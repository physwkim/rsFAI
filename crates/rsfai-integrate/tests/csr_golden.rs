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
use rsfai_integrate::{
    build_bbox_csr_1d, build_bbox_csr_2d, build_full_csr_1d, build_full_csr_2d, csr_integrate1d,
    csr_integrate2d, Bbox2dBounds, Csr,
};

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets")
}

fn dataset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    if let Ok(rd) = std::fs::read_dir(datasets_root()) {
        for e in rd.flatten() {
            // Cython golden datasets only: skip the Phase-2 OpenCL datasets,
            // which carry an `opencl_params.json` and a reduced manifest with no
            // cython intermediates. `rsfai-opencl`'s own golden test owns those.
            if e.path().join("manifest.json").exists()
                && !e.path().join("opencl_params.json").exists()
            {
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
        // The 2D bbox-CSR dataset shares the ("bbox","csr") method tuple; this
        // 1D test must skip it (it carries npt_rad/npt_azim, not npt) — covered
        // by `bbox_csr_2d_build_and_apply_bit_exact`.
        if cfg["dim"].as_u64().unwrap_or(1) != 1 {
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

/// Tier-A validation of the no-split CSR 1D path: pyFAI's `("no", "csr", …)`
/// reuses the same `HistoBBox1d` as bbox-CSR, constructed with `delta = None`,
/// which gates `do_split = False` so every unmasked in-range pixel becomes a
/// single coef-1.0 entry at its center bin. The Rust mirror is
/// [`build_bbox_csr_1d`] called with `delta_pos0 = None`; the built CSR and the
/// applied output (`csr_integrate1d`, fed the *golden* CSR + *golden* preproc)
/// must be bit-exact vs golden. Inputs and gate match
/// `bbox_csr_build_and_apply_bit_exact`; only the absence of `pos0_delta`
/// differs.
#[test]
fn nosplit_csr_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // Only the no-split CSR path: method == ("no", "csr", _).
        let is_no_csr = cfg["method"]
            .as_array()
            .map(|m| m.len() >= 3 && m[0].as_str() == Some("no") && m[1].as_str() == Some("csr"))
            .unwrap_or(false);
        if !is_no_csr {
            continue;
        }
        // 1D only; the 2D no-csr dataset shares the ("no","csr") tuple but carries
        // npt_rad/npt_azim — covered by `nosplit_csr_2d_build_and_apply_bit_exact`.
        if cfg["dim"].as_u64().unwrap_or(1) != 1 {
            continue;
        }

        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        // Standard radial units (q, 2θ) cannot be negative -> allow_pos0_neg=false.
        let allow_pos0_neg = false;

        // --- Build: identical center array, no delta (do_split=False) ---
        let pos0 = vec_f64(&dir, "pos0_center_unscaled.npy");
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let (built, bin_centers) = build_bbox_csr_1d(&pos0, None, Some(&mask), npt, allow_pos0_neg);

        let g_data = vec_f32(&dir, "csr_data.npy");
        let g_indices = vec_i32(&dir, "csr_indices.npy");
        let g_indptr = vec_i32(&dir, "csr_indptr.npy");

        let r_data = compare_f32(&built.data, &g_data);
        let indices_eq = built.indices == g_indices;
        let indptr_eq = built.indptr == g_indptr;
        eprintln!(
            "{}: no-split csr nnz={} (golden {}) | data mism={} | indices eq={} | indptr eq={}",
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

        eprintln!(
            "{}: no-split build + apply all fields bit-exact",
            manifest.dataset
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no no-split-csr 1D golden datasets found; run golden/gen_golden.py"
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
        // Future-proof against a 2D full-CSR dataset sharing the ("full","csr")
        // method tuple: this 1D test reads npt, which a 2D config does not carry.
        if cfg["dim"].as_u64().unwrap_or(1) != 1 {
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

/// Tier-A validation of the 2D bbox→CSR path: the built 2D LUT
/// (`build_bbox_csr_2d` ← `calc_lut_2d`) and the applied output
/// (`csr_integrate2d` ← `integrate_ng`'s 2D return) must be bit-exact vs golden,
/// fed the identical inputs pyFAI's `HistoBBox2d` received: the unscaled radial
/// center/half-width (`pos0_center_unscaled` / `pos0_delta`) and the radian
/// azimuthal center/half-width (`chi_center` / `chi_delta`), confirmed equal to
/// the engine's `cpos0/dpos0/cpos1/dpos1`. The build is checked against the
/// golden `csr_*`; the apply is fed the *golden* CSR + *golden* preproc rows.
/// The reported axes are `radial * unit_scale` and `azimuthal * azim_scale`
/// (CHI_DEG = 180/π). `HistoBBox2d` takes the constructor default
/// `chiDiscAtPi=True` (common.py does not forward it) and `pos1_period =
/// CHI_DEG.period`.
#[test]
fn bbox_csr_2d_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // 2D bbox CSR path: dim == 2, method == ("bbox","csr",_).
        if cfg["dim"].as_u64().unwrap_or(1) != 2 {
            continue;
        }
        let is_bbox_csr = cfg["method"]
            .as_array()
            .map(|m| m.len() >= 3 && m[0].as_str() == Some("bbox") && m[1].as_str() == Some("csr"))
            .unwrap_or(false);
        if !is_bbox_csr {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        // Standard radial unit (q) cannot be negative -> allow_pos0_neg=false.
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi,
            pos1_period,
        };

        // --- Build: the per-pixel arrays HistoBBox2d received ---
        let pos0 = vec_f64(&dir, "pos0_center_unscaled.npy");
        let dpos0 = vec_f64(&dir, "pos0_delta.npy");
        let pos1 = vec_f64(&dir, "chi_center.npy");
        let dpos1 = vec_f64(&dir, "chi_delta.npy");
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let (built, bin_centers0, bin_centers1) = build_bbox_csr_2d(
            &pos0,
            Some(&dpos0),
            &pos1,
            Some(&dpos1),
            Some(&mask),
            (npt_rad, npt_azim),
            &bounds,
        );

        let g_data = vec_f32(&dir, "csr_data.npy");
        let g_indices = vec_i32(&dir, "csr_indices.npy");
        let g_indptr = vec_i32(&dir, "csr_indptr.npy");

        let r_data = compare_f32(&built.data, &g_data);
        let indices_eq = built.indices == g_indices;
        let indptr_eq = built.indptr == g_indptr;
        eprintln!(
            "{}: 2D csr nnz={} (golden {}) | data mism={} | indices eq={} | indptr eq={}",
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
        let out = csr_integrate2d(
            &golden_csr,
            &prep,
            bin_centers0,
            bin_centers1,
            error_model,
            0.0,
        );

        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = vec_f64(&dir, "out_radial.npy");
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );
        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = vec_f64(&dir, "out_azimuthal.npy");
        let r_azim = compare_f64(&scaled_azim, &g_azim);
        assert!(
            r_azim.is_bit_exact(),
            "{}: out_azimuthal not bit-exact: {r_azim:?}",
            manifest.dataset
        );

        // f64 accumulator fields (the CSR sums are exposed at full precision).
        let check_f64 = |file: &str, actual: &[f64], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_npy_f64(&p)
                .unwrap()
                .as_slice()
                .expect("contiguous")
                .to_vec();
            let r = compare_f64(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };
        let check_f32 = |file: &str, actual: &[f32], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_npy_f32(&p)
                .unwrap()
                .as_slice()
                .expect("contiguous")
                .to_vec();
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };

        check_f64("out_sum_signal.npy", &out.signal, true);
        check_f64("out_sum_normalization.npy", &out.normalization, true);
        check_f64("out_count.npy", &out.count, true);
        check_f32("out_intensity.npy", &out.intensity, true);

        // Variance-family fields: exposed only when do_variance (error_model != No).
        let mut em_fields = 0;
        em_fields += i32::from(check_f64("out_sum_variance.npy", &out.variance, false));
        em_fields += i32::from(check_f64("out_sum_normalization2.npy", &out.norm_sq, false));
        em_fields += i32::from(check_f32("out_std.npy", &out.std, false));
        em_fields += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em_fields += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em_fields > 0,
                "{}: error-model dataset exposed no variance/std/sem fields to validate",
                manifest.dataset
            );
        }

        eprintln!(
            "{}: 2D build + apply all fields bit-exact (radial ulp={}, azim ulp={}, {em_fields} error-model fields)",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D bbox-csr golden datasets found; run golden/gen_golden.py"
    );
}

/// Tier-A validation of the no-split CSR 2D path: pyFAI's `("no", "csr", …)`
/// reuses the same `HistoBBox2d` as bbox-CSR, constructed with `delta = None`,
/// gating `do_split = False` so every unmasked in-range pixel becomes a single
/// coef-1.0 entry at its `(radial, azimuthal)` center cell. The Rust mirror is
/// [`build_bbox_csr_2d`] called with `delta_pos0 = delta_pos1 = None`; the built
/// CSR and the applied output (`csr_integrate2d`, fed the *golden* CSR + *golden*
/// preproc) must be bit-exact vs golden. Inputs and gate match
/// `bbox_csr_2d_build_and_apply_bit_exact`; only the absence of the half-width
/// arrays differs.
#[test]
fn nosplit_csr_2d_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // 2D no-split CSR path: dim == 2, method == ("no","csr",_).
        if cfg["dim"].as_u64().unwrap_or(1) != 2 {
            continue;
        }
        let is_no_csr = cfg["method"]
            .as_array()
            .map(|m| m.len() >= 3 && m[0].as_str() == Some("no") && m[1].as_str() == Some("csr"))
            .unwrap_or(false);
        if !is_no_csr {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        // Standard radial unit (q) cannot be negative -> allow_pos0_neg=false.
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi,
            pos1_period,
        };

        // --- Build: center arrays only, no half-widths (do_split=False) ---
        let pos0 = vec_f64(&dir, "pos0_center_unscaled.npy");
        let pos1 = vec_f64(&dir, "chi_center.npy");
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let (built, bin_centers0, bin_centers1) = build_bbox_csr_2d(
            &pos0,
            None,
            &pos1,
            None,
            Some(&mask),
            (npt_rad, npt_azim),
            &bounds,
        );

        let g_data = vec_f32(&dir, "csr_data.npy");
        let g_indices = vec_i32(&dir, "csr_indices.npy");
        let g_indptr = vec_i32(&dir, "csr_indptr.npy");

        let r_data = compare_f32(&built.data, &g_data);
        let indices_eq = built.indices == g_indices;
        let indptr_eq = built.indptr == g_indptr;
        eprintln!(
            "{}: 2D no-split csr nnz={} (golden {}) | data mism={} | indices eq={} | indptr eq={}",
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
        let out = csr_integrate2d(
            &golden_csr,
            &prep,
            bin_centers0,
            bin_centers1,
            error_model,
            0.0,
        );

        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = vec_f64(&dir, "out_radial.npy");
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );
        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = vec_f64(&dir, "out_azimuthal.npy");
        let r_azim = compare_f64(&scaled_azim, &g_azim);
        assert!(
            r_azim.is_bit_exact(),
            "{}: out_azimuthal not bit-exact: {r_azim:?}",
            manifest.dataset
        );

        // f64 accumulator fields (the CSR sums are exposed at full precision).
        let check_f64 = |file: &str, actual: &[f64], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_npy_f64(&p)
                .unwrap()
                .as_slice()
                .expect("contiguous")
                .to_vec();
            let r = compare_f64(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };
        let check_f32 = |file: &str, actual: &[f32], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_npy_f32(&p)
                .unwrap()
                .as_slice()
                .expect("contiguous")
                .to_vec();
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };

        check_f64("out_sum_signal.npy", &out.signal, true);
        check_f64("out_sum_normalization.npy", &out.normalization, true);
        check_f64("out_count.npy", &out.count, true);
        check_f32("out_intensity.npy", &out.intensity, true);

        // Variance-family fields: exposed only when do_variance (error_model != No).
        let mut em_fields = 0;
        em_fields += i32::from(check_f64("out_sum_variance.npy", &out.variance, false));
        em_fields += i32::from(check_f64("out_sum_normalization2.npy", &out.norm_sq, false));
        em_fields += i32::from(check_f32("out_std.npy", &out.std, false));
        em_fields += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em_fields += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em_fields > 0,
                "{}: error-model dataset exposed no variance/std/sem fields to validate",
                manifest.dataset
            );
        }

        eprintln!(
            "{}: 2D no-split build + apply all fields bit-exact (radial ulp={}, azim ulp={}, {em_fields} error-model fields)",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D no-split-csr golden datasets found; run golden/gen_golden.py"
    );
}

/// Tier-A validation of the 2D full pixel-splitting CSR path: the built 2D LUT
/// (`build_full_csr_2d` ← `splitpixel_common.calc_lut_2d`, sweeping each pixel's
/// recentered corners with `_integrate2d`) and the applied output
/// (`csr_integrate2d`) must be bit-exact vs golden, fed the identical corner array
/// pyFAI's `FullSplitIntegrator` received (`corners`, f32 upcast to f64 = the
/// engine's `pos`). The build is checked against the golden `csr_*`; the apply is
/// fed the *golden* CSR + *golden* preproc rows. Unlike the bbox-2D and 1D-full
/// paths, `common.py` forwards `chiDiscAtPi` and `pos1_period = unit1.period`
/// (360) to `FullSplitCSR_2d`; both come from the manifest.
#[test]
fn full_csr_2d_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // 2D full-split CSR path: dim == 2, method == ("full","csr",_).
        if cfg["dim"].as_u64().unwrap_or(1) != 2 {
            continue;
        }
        let is_full_csr = cfg["method"]
            .as_array()
            .map(|m| m.len() >= 3 && m[0].as_str() == Some("full") && m[1].as_str() == Some("csr"))
            .unwrap_or(false);
        if !is_full_csr {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        // Standard radial unit (q) cannot be negative -> allow_pos0_neg=false.
        let allow_pos0_neg = false;

        // --- Build: the corner array pyFAI's FullSplitIntegrator received ---
        let corners = corners_f64(&dir);
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let (built, bin_centers0, bin_centers1) = build_full_csr_2d(
            &corners,
            Some(&mask),
            (npt_rad, npt_azim),
            allow_pos0_neg,
            chi_disc_at_pi,
            pos1_period,
        );

        let g_data = vec_f32(&dir, "csr_data.npy");
        let g_indices = vec_i32(&dir, "csr_indices.npy");
        let g_indptr = vec_i32(&dir, "csr_indptr.npy");

        let r_data = compare_f32(&built.data, &g_data);
        let indices_eq = built.indices == g_indices;
        let indptr_eq = built.indptr == g_indptr;
        eprintln!(
            "{}: 2D full csr nnz={} (golden {}) | data mism={} | indices eq={} | indptr eq={}",
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
        let out = csr_integrate2d(
            &golden_csr,
            &prep,
            bin_centers0,
            bin_centers1,
            error_model,
            0.0,
        );

        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = vec_f64(&dir, "out_radial.npy");
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );
        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = vec_f64(&dir, "out_azimuthal.npy");
        let r_azim = compare_f64(&scaled_azim, &g_azim);
        assert!(
            r_azim.is_bit_exact(),
            "{}: out_azimuthal not bit-exact: {r_azim:?}",
            manifest.dataset
        );

        let check_f64 = |file: &str, actual: &[f64], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_npy_f64(&p)
                .unwrap()
                .as_slice()
                .expect("contiguous")
                .to_vec();
            let r = compare_f64(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };
        let check_f32 = |file: &str, actual: &[f32], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_npy_f32(&p)
                .unwrap()
                .as_slice()
                .expect("contiguous")
                .to_vec();
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };

        check_f64("out_sum_signal.npy", &out.signal, true);
        check_f64("out_sum_normalization.npy", &out.normalization, true);
        check_f64("out_count.npy", &out.count, true);
        check_f32("out_intensity.npy", &out.intensity, true);

        // Variance-family fields: exposed only when do_variance (error_model != No).
        let mut em_fields = 0;
        em_fields += i32::from(check_f64("out_sum_variance.npy", &out.variance, false));
        em_fields += i32::from(check_f64("out_sum_normalization2.npy", &out.norm_sq, false));
        em_fields += i32::from(check_f32("out_std.npy", &out.std, false));
        em_fields += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em_fields += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em_fields > 0,
                "{}: error-model dataset exposed no variance/std/sem fields to validate",
                manifest.dataset
            );
        }

        eprintln!(
            "{}: 2D full build + apply all fields bit-exact (radial ulp={}, azim ulp={}, {em_fields} error-model fields)",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D full-csr golden datasets found; run golden/gen_golden.py"
    );
}
