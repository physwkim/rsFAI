//! Validate the Rust CSR-NG OpenCL orchestration against pyFAI's OpenCL golden.
//!
//! For every `golden/datasets/*` carrying an `opencl_params.json` (produced by
//! `golden/gen_golden_opencl.py`), this feeds the *same* inputs pyFAI fed its
//! GPU — raw image, correction arrays, and the exact CSR matrix — through
//! pyFAI's own embedded kernels via [`rsfai_opencl::integrate1d_csr`], then
//! compares each output field against the golden.
//!
//! Gate: relative error <= 1e-6 (the Phase-2 tolerance for the GPU doubleword
//! reduction). On the Apple M4 Pro the deterministic same-device/same-kernel
//! path is expected to be bit-exact; the test reports which fields actually
//! matched bit-for-bit so any drift from that is visible.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use rsfai_core::compare::compare_f32;
use rsfai_core::golden::{load_npy_f32, load_npy_f64, load_npy_i32, load_npy_i8};
use rsfai_opencl::{create_queue, default_context, program, Corrections4aArgs, CsrInputs};

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

#[derive(Deserialize)]
struct Corr {
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

#[derive(Deserialize)]
struct OpenclParams {
    corrections4a: Corr,
    wg_min: usize,
    bins: usize,
    image_size: usize,
    empty: f32,
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
/// Load an f64 `.npy` and cast to f32 the way pyFAI's `send_buffer` does
/// (`numpy.ascontiguousarray(data, float32)`, i.e. round-to-nearest-even).
fn vec_f64_as_f32(p: PathBuf) -> Vec<f32> {
    load_npy_f64(p)
        .expect("f64 npy")
        .iter()
        .map(|&v| v as f32)
        .collect()
}

/// Optional correction array: load as f32 only if the file exists (absent ⇒ the
/// matching `do_*` flag is 0, so the kernel ignores it). pyFAI stores solid
/// angle / polarization as f64; cast to f32 like the GPU upload.
fn opt_f32(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let p = dir.join(format!("{name}.npy"));
    p.exists().then(|| vec_f64_as_f32(p))
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

#[test]
fn csr_opencl_matches_pyfai_golden() {
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
        let mask = vec_i8(dir.join("mask.npy"));
        let solidangle = opt_f32(dir, "solidangle");
        let polarization = opt_f32(dir, "polarization");

        let csr_data = vec_f32(dir.join("csr_data.npy"));
        let csr_indices = vec_i32(dir.join("csr_indices.npy"));
        let csr_indptr = vec_i32(dir.join("csr_indptr.npy"));

        let c = &params.corrections4a;
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
        let inputs = CsrInputs {
            image_i32: &image,
            variance: None,
            dark: None,
            dark_variance: None,
            flat: None,
            solidangle: solidangle.as_deref(),
            polarization: polarization.as_deref(),
            absorption: None,
            mask: Some(&mask),
            csr_data: &csr_data,
            csr_indices: &csr_indices,
            csr_indptr: &csr_indptr,
        };

        let prog = program::build_csr_program(&context, params.bins, params.image_size)
            .unwrap_or_else(|e| panic!("{dataset}: build CSR program: {e}"));
        let res = rsfai_opencl::integrate1d_csr(
            &context,
            &queue,
            &prog,
            &inputs,
            &corr,
            params.empty,
            params.wg_min,
        )
        .unwrap_or_else(|e| panic!("{dataset}: integrate1d_csr: {e}"));

        // pyFAI Integrate1dtpl field mapping (azim_csr.integrate_ng):
        //   intensity=averint, sigma=sem, std=std, sem=sem,
        //   signal=merged0, variance=merged2, normalization=merged4,
        //   count=merged6, norm_sq=merged7.
        let golden = |name: &str| vec_f32(dir.join(format!("out_{name}.npy")));
        let fields: &[(&str, &[f32], Vec<f32>)] = &[
            ("intensity", &res.intensity, golden("intensity")),
            ("sigma", &res.sem, golden("sigma")),
            ("std", &res.std, golden("std")),
            ("sem", &res.sem, golden("sem")),
            ("sum_signal", &res.signal, golden("sum_signal")),
            ("sum_variance", &res.variance, golden("sum_variance")),
            (
                "sum_normalization",
                &res.normalization,
                golden("sum_normalization"),
            ),
            ("count", &res.count, golden("count")),
            (
                "sum_normalization2",
                &res.norm_sq,
                golden("sum_normalization2"),
            ),
        ];
        let mut bit_exact_fields = 0usize;
        for (name, actual, gold) in fields {
            if check_field(&dataset, name, actual, gold) {
                bit_exact_fields += 1;
            }
        }
        eprintln!(
            "{dataset}: {}/{} fields bit-exact (rest within rel<= {:e})",
            bit_exact_fields,
            fields.len(),
            REL_TOL
        );
        checked += 1;
    }
    assert!(checked > 0, "no OpenCL datasets validated");
}
