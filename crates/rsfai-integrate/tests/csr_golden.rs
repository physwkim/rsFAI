//! Tier-A validation of the bbox→CSR 1D path: the built CSR matrix
//! (`build_bbox_csr_1d`) and the applied output (`csr_integrate1d`) must be
//! bit-exact vs golden, fed the identical inputs pyFAI used.
//!
//! The two halves are validated separately: the build against the golden
//! `csr_data`/`csr_indices`/`csr_indptr`, and the apply against the golden
//! `out_*` while fed the *golden* CSR + the *golden* preproc rows (so an apply
//! failure can't be masked by a build failure). The engine works in unscaled
//! radial space (`pos0_center_unscaled`, `pos0_delta`); the reported position is
//! `bin_centers * unit_scale`.

use std::path::PathBuf;

use std::f64::consts::PI;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, load_npy_i32, load_npy_i8};
use rsfai_integrate::{build_bbox_csr_1d, build_full_csr_1d, csr_integrate1d, Csr};

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn dataset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(datasets_root()) {
        for e in rd.flatten() {
            if e.path().join("manifest.json").exists() {
                dirs.push(e.path());
            }
        }
    }
    dirs.sort();
    dirs
}

fn error_model_from_code(code: i64) -> ErrorModel {
    match code {
        0 => ErrorModel::No,
        1 => ErrorModel::Variance,
        2 => ErrorModel::Poisson,
        3 => ErrorModel::Azimuthal,
        other => panic!("unknown error_model_code {other}"),
    }
}

fn vec_f64(dir: &std::path::Path, file: &str) -> Vec<f64> {
    load_npy_f64(dir.join(file))
        .unwrap_or_else(|_| panic!("load {file}"))
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn vec_f32(dir: &std::path::Path, file: &str) -> Vec<f32> {
    load_npy_f32(dir.join(file))
        .unwrap_or_else(|_| panic!("load {file}"))
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn vec_i32(dir: &std::path::Path, file: &str) -> Vec<i32> {
    load_npy_i32(dir.join(file))
        .unwrap_or_else(|_| panic!("load {file}"))
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

#[test]
fn bbox_csr_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // Only the bbox CSR path: method == ("bbox", "csr", _).
        let is_bbox_csr = cfg["method"]
            .as_array()
            .map(|m| m.len() >= 3 && m[0].as_str() == Some("bbox") && m[1].as_str() == Some("csr"))
            .unwrap_or(false);
        if !is_bbox_csr {
            continue;
        }

        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        // Standard radial units (q, 2θ) cannot be negative; the integrator binds
        // allow_pos0_neg = false for them.
        let allow_pos0_neg = false;

        // --- Build: identical inputs pyFAI's SplitBBoxIntegrator received ---
        let pos0 = vec_f64(&dir, "pos0_center_unscaled.npy");
        let dpos0 = vec_f64(&dir, "pos0_delta.npy");
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let (built, bin_centers) =
            build_bbox_csr_1d(&pos0, Some(&dpos0), Some(&mask), npt, allow_pos0_neg);

        let g_data = vec_f32(&dir, "csr_data.npy");
        let g_indices = vec_i32(&dir, "csr_indices.npy");
        let g_indptr = vec_i32(&dir, "csr_indptr.npy");

        let r_data = compare_f32(&built.data, &g_data);
        let indices_eq = built.indices == g_indices;
        let indptr_eq = built.indptr == g_indptr;
        eprintln!(
            "{}: csr nnz={} (golden {}) | data mism={} | indices eq={} | indptr eq={}",
            manifest.dataset,
            built.data.len(),
            g_data.len(),
            r_data.bit_mismatches,
            indices_eq,
            indptr_eq,
        );
        assert!(indptr_eq, "{}: csr_indptr mismatch", manifest.dataset);
        assert!(indices_eq, "{}: csr_indices mismatch", manifest.dataset);
        assert!(
            r_data.is_bit_exact(),
            "{}: csr_data not bit-exact: {r_data:?}",
            manifest.dataset
        );

        // --- Apply: feed the GOLDEN CSR + GOLDEN preproc rows ---
        let golden_csr = Csr {
            data: g_data,
            indices: g_indices,
            indptr: g_indptr,
        };
        let prep = vec_f32(&dir, "preproc.npy");
        let out = csr_integrate1d(&golden_csr, &prep, bin_centers, error_model, 0.0);

        // position * unit_scale == out_radial (f64). pyFAI: intpl.position * unit.scale.
        let scaled: Vec<f64> = out.position.iter().map(|&p| p * unit_scale).collect();
        let g_radial = vec_f64(&dir, "out_radial.npy");
        let r_radial = compare_f64(&scaled, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial (position*scale) not bit-exact: {r_radial:?}",
            manifest.dataset
        );

        // f64 accumulator fields (pyFAI exposes the CSR sums at full precision).
        let cmp64 = |file: &str, actual: &[f64]| {
            let g = vec_f64(&dir, file);
            let r = compare_f64(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
        };
        cmp64("out_sum_signal.npy", &out.sum_signal);
        cmp64("out_sum_normalization.npy", &out.sum_normalization);
        cmp64("out_sum_normalization2.npy", &out.sum_norm_sq);
        cmp64("out_count.npy", &out.count);
        cmp64("out_sum_variance.npy", &out.sum_variance);

        // f32 derived fields.
        let cmp32 = |file: &str, actual: &[f32]| {
            let g = vec_f32(&dir, file);
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
        };
        cmp32("out_intensity.npy", &out.intensity);
        cmp32("out_sigma.npy", &out.sigma);
        cmp32("out_std.npy", &out.std);
        cmp32("out_sem.npy", &out.sem);

        eprintln!("{}: build + apply all fields bit-exact", manifest.dataset);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no bbox-csr golden datasets found; run golden/gen_golden.py"
    );
}

/// The `corner` array is stored f32 (`(H, W, 4, 2)`, C-order); the engine upcasts
/// it to f64 (`ascontiguousarray(pos, dtype=position_d)`) before binning. Load it
/// flat and widen — the (H·W, 4, 2) flattening matches pixel index `row*W + col`.
fn corners_f64(dir: &std::path::Path) -> Vec<f64> {
    load_npy_f32(dir.join("corners.npy"))
        .expect("load corners.npy")
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|&v| v as f64)
        .collect()
}

#[test]
fn full_split_csr_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // Only the full-split CSR path: method == ("full", "csr", _).
        let is_full_csr = cfg["method"]
            .as_array()
            .map(|m| m.len() >= 3 && m[0].as_str() == Some("full") && m[1].as_str() == Some("csr"))
            .unwrap_or(false);
        if !is_full_csr {
            continue;
        }

        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        // Standard radial units (q, 2θ) cannot be negative -> allow_pos0_neg=false.
        // The 1D full-split CSR setup (common.py) does NOT forward ai.chiDiscAtPi
        // or pos1_period to FullSplitCSR_1d, so they take the constructor
        // defaults: chiDiscAtPi=True, pos1_period=2π.
        let allow_pos0_neg = false;

        // --- Build: the corner array pyFAI's FullSplitIntegrator received ---
        let corners = corners_f64(&dir);
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let (built, bin_centers) =
            build_full_csr_1d(&corners, Some(&mask), npt, allow_pos0_neg, true, 2.0 * PI);

        let g_data = vec_f32(&dir, "csr_data.npy");
        let g_indices = vec_i32(&dir, "csr_indices.npy");
        let g_indptr = vec_i32(&dir, "csr_indptr.npy");

        let r_data = compare_f32(&built.data, &g_data);
        let indices_eq = built.indices == g_indices;
        let indptr_eq = built.indptr == g_indptr;
        eprintln!(
            "{}: csr nnz={} (golden {}) | data mism={} | indices eq={} | indptr eq={}",
            manifest.dataset,
            built.data.len(),
            g_data.len(),
            r_data.bit_mismatches,
            indices_eq,
            indptr_eq,
        );
        assert!(indptr_eq, "{}: csr_indptr mismatch", manifest.dataset);
        assert!(indices_eq, "{}: csr_indices mismatch", manifest.dataset);
        assert!(
            r_data.is_bit_exact(),
            "{}: csr_data not bit-exact: {r_data:?}",
            manifest.dataset
        );

        // --- Apply: feed the GOLDEN CSR + GOLDEN preproc rows ---
        let golden_csr = Csr {
            data: g_data,
            indices: g_indices,
            indptr: g_indptr,
        };
        let prep = vec_f32(&dir, "preproc.npy");
        let out = csr_integrate1d(&golden_csr, &prep, bin_centers, error_model, 0.0);

        let scaled: Vec<f64> = out.position.iter().map(|&p| p * unit_scale).collect();
        let g_radial = vec_f64(&dir, "out_radial.npy");
        let r_radial = compare_f64(&scaled, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial (position*scale) not bit-exact: {r_radial:?}",
            manifest.dataset
        );

        let cmp64 = |file: &str, actual: &[f64]| {
            let g = vec_f64(&dir, file);
            let r = compare_f64(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
        };
        cmp64("out_sum_signal.npy", &out.sum_signal);
        cmp64("out_sum_normalization.npy", &out.sum_normalization);
        cmp64("out_sum_normalization2.npy", &out.sum_norm_sq);
        cmp64("out_count.npy", &out.count);
        cmp64("out_sum_variance.npy", &out.sum_variance);

        let cmp32 = |file: &str, actual: &[f32]| {
            let g = vec_f32(&dir, file);
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
        };
        cmp32("out_intensity.npy", &out.intensity);
        cmp32("out_sigma.npy", &out.sigma);
        cmp32("out_std.npy", &out.std);
        cmp32("out_sem.npy", &out.sem);

        eprintln!("{}: build + apply all fields bit-exact", manifest.dataset);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no full-csr golden datasets found; run golden/gen_golden.py"
    );
}
