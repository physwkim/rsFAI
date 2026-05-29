//! Tier-A validation of the LUT paths (`("no"|"bbox"|"full", "lut", "cython")`,
//! 1D + 2D). The built dense LUT (densified CSR via `to_lut`) is validated against
//! the golden `lut_idx`/`lut_coef` `(n_bins, lut_size)` matrix, and the applied
//! output (`lut_integrate1d`/`lut_integrate2d`, fed the *golden* LUT + *golden*
//! preproc rows so a build failure can't mask an apply failure) against the golden
//! `out_*`. Every field is asserted bit-exact: the build matches `SparseBuilder.
//! to_lut`, and the per-bin gather reproduces pyFAI's `LutIntegrator.integrate_ng`
//! bit-for-bit.

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, load_npy_i32, load_npy_i8};
use rsfai_integrate::{
    build_bbox_lut_1d, build_bbox_lut_2d, build_full_lut_1d, build_full_lut_2d, lut_integrate1d,
    lut_integrate2d, Bbox2dBounds, Lut,
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
            let dir = e.path();
            if dir.join("manifest.json").exists() && !dir.join("opencl_params.json").exists() {
                // A user radial_range/azimuth_range overrides the binning
                // boundaries, but these per-kernel tests drive the raw build/
                // histogram kernels on the full data extent (no override), so a
                // range-built golden cannot match. The range path is validated
                // end-to-end by `dropin_golden.rs`.
                let ranged = load_manifest(dir.join("manifest.json"))
                    .map(|m| {
                        !m.config["radial_range"].is_null() || !m.config["azimuth_range"].is_null()
                    })
                    .unwrap_or(false);
                if !ranged {
                    dirs.push(dir);
                }
            }
        }
    }
    dirs.sort();
    dirs
}

fn error_model_from_code(code: i64) -> ErrorModel {
    ErrorModel::from_code(code as i32).unwrap_or_else(|| panic!("unknown error_model_code {code}"))
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

fn mask(dir: &std::path::Path) -> Vec<i8> {
    load_npy_i8(dir.join("mask.npy"))
        .expect("mask")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

/// The `(npix, 4, 2)` corner array, stored f32, upcast to f64 (as pyFAI's
/// `FullSplitIntegrator` does).
fn corners_f64(dir: &std::path::Path) -> Vec<f64> {
    load_npy_f32(dir.join("corners.npy"))
        .expect("corners")
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// Does this dataset's method tuple match `(split, "lut")` at the given dim?
fn is_lut(cfg: &serde_json::Value, split: &str, dim: u64) -> bool {
    let matches = cfg["method"]
        .as_array()
        .map(|m| m.len() >= 3 && m[0].as_str() == Some(split) && m[1].as_str() == Some("lut"))
        .unwrap_or(false);
    matches && cfg["dim"].as_u64().unwrap_or(1) == dim
}

/// Load the golden dense LUT matrix and reconstruct a [`Lut`]. `n_bins` is the
/// number of output bins (`npt` for 1D, `bins0·bins1` for 2D); `lut_size` is
/// inferred from the flat lengths.
fn golden_lut(dir: &std::path::Path, n_bins: usize) -> Lut {
    let idx = vec_i32(dir, "lut_idx.npy");
    let coef = vec_f32(dir, "lut_coef.npy");
    assert_eq!(idx.len(), coef.len(), "lut_idx/lut_coef length mismatch");
    let lut_size = if n_bins == 0 { 0 } else { coef.len() / n_bins };
    assert_eq!(
        n_bins * lut_size,
        coef.len(),
        "lut matrix not (n_bins, lut_size)"
    );
    Lut::new(coef, idx, lut_size)
}

/// Compare the built dense LUT against the golden, then apply the *golden* LUT +
/// *golden* preproc and assert every 1D output field bit-exact.
fn validate_lut_1d(
    dir: &std::path::Path,
    dataset: &str,
    cfg: &serde_json::Value,
    built: Lut,
    bin_centers: Vec<f64>,
) {
    let unit = cfg["unit"].as_str().unwrap_or("");
    assert_eq!(unit, "q_nm^-1", "{dataset}: 1D LUT golden assumes q_nm^-1");
    let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
    let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
    let n_bins = bin_centers.len();

    let golden = golden_lut(dir, n_bins);
    let r_coef = compare_f32(&built.coef, &golden.coef);
    let idx_eq = built.idx == golden.idx;
    eprintln!(
        "{dataset}: lut_size={} (golden {}) | coef mism={} | idx eq={}",
        built.lut_size, golden.lut_size, r_coef.bit_mismatches, idx_eq,
    );
    assert_eq!(
        built.lut_size, golden.lut_size,
        "{dataset}: lut_size mismatch"
    );
    assert!(idx_eq, "{dataset}: lut_idx mismatch");
    assert!(
        r_coef.is_bit_exact(),
        "{dataset}: lut_coef not bit-exact: {r_coef:?}"
    );

    let prep = vec_f32(dir, "preproc.npy");
    let out = lut_integrate1d(&golden, &prep, bin_centers, error_model, 0.0);

    let scaled: Vec<f64> = out.position.iter().map(|&p| p * unit_scale).collect();
    let g_radial = vec_f64(dir, "out_radial.npy");
    let r_radial = compare_f64(&scaled, &g_radial);
    assert!(
        r_radial.is_bit_exact(),
        "{dataset}: out_radial (position*scale) not bit-exact: {r_radial:?}"
    );

    let cmp64 = |file: &str, actual: &[f64]| {
        let g = vec_f64(dir, file);
        let r = compare_f64(actual, &g);
        assert!(r.is_bit_exact(), "{dataset}: {file} not bit-exact: {r:?}");
    };
    cmp64("out_sum_signal.npy", &out.sum_signal);
    cmp64("out_sum_normalization.npy", &out.sum_normalization);
    cmp64("out_sum_normalization2.npy", &out.sum_norm_sq);
    cmp64("out_count.npy", &out.count);
    cmp64("out_sum_variance.npy", &out.sum_variance);

    let cmp32 = |file: &str, actual: &[f32]| {
        let g = vec_f32(dir, file);
        let r = compare_f32(actual, &g);
        assert!(r.is_bit_exact(), "{dataset}: {file} not bit-exact: {r:?}");
    };
    cmp32("out_intensity.npy", &out.intensity);
    cmp32("out_sigma.npy", &out.sigma);
    cmp32("out_std.npy", &out.std);
    cmp32("out_sem.npy", &out.sem);

    eprintln!("{dataset}: 1D LUT build + apply all fields bit-exact");
}

/// Compare the built 2D dense LUT against the golden, then apply the *golden* LUT
/// + *golden* preproc and assert every 2D output field bit-exact.
fn validate_lut_2d(
    dir: &std::path::Path,
    dataset: &str,
    cfg: &serde_json::Value,
    built: Lut,
    bin_centers0: Vec<f64>,
    bin_centers1: Vec<f64>,
) {
    let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
    let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
    let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
    let n_bins = bin_centers0.len() * bin_centers1.len();

    let golden = golden_lut(dir, n_bins);
    let r_coef = compare_f32(&built.coef, &golden.coef);
    let idx_eq = built.idx == golden.idx;
    eprintln!(
        "{dataset}: 2D lut_size={} (golden {}) | coef mism={} | idx eq={}",
        built.lut_size, golden.lut_size, r_coef.bit_mismatches, idx_eq,
    );
    assert_eq!(
        built.lut_size, golden.lut_size,
        "{dataset}: lut_size mismatch"
    );
    assert!(idx_eq, "{dataset}: lut_idx mismatch");
    assert!(
        r_coef.is_bit_exact(),
        "{dataset}: lut_coef not bit-exact: {r_coef:?}"
    );

    let prep = vec_f32(dir, "preproc.npy");
    let out = lut_integrate2d(&golden, &prep, bin_centers0, bin_centers1, error_model, 0.0);

    let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
    let g_radial = vec_f64(dir, "out_radial.npy");
    let r_radial = compare_f64(&scaled_rad, &g_radial);
    assert!(
        r_radial.is_bit_exact(),
        "{dataset}: out_radial not bit-exact: {r_radial:?}"
    );
    let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
    let g_azim = vec_f64(dir, "out_azimuthal.npy");
    let r_azim = compare_f64(&scaled_azim, &g_azim);
    assert!(
        r_azim.is_bit_exact(),
        "{dataset}: out_azimuthal not bit-exact: {r_azim:?}"
    );

    let cmp64 = |file: &str, actual: &[f64]| {
        let g = vec_f64(dir, file);
        let r = compare_f64(actual, &g);
        assert!(r.is_bit_exact(), "{dataset}: {file} not bit-exact: {r:?}");
    };
    cmp64("out_sum_signal.npy", &out.signal);
    cmp64("out_sum_normalization.npy", &out.normalization);
    cmp64("out_sum_normalization2.npy", &out.norm_sq);
    cmp64("out_count.npy", &out.count);
    cmp64("out_sum_variance.npy", &out.variance);

    let cmp32 = |file: &str, actual: &[f32]| {
        let g = vec_f32(dir, file);
        let r = compare_f32(actual, &g);
        assert!(r.is_bit_exact(), "{dataset}: {file} not bit-exact: {r:?}");
    };
    cmp32("out_intensity.npy", &out.intensity);
    cmp32("out_sigma.npy", &out.sigma);
    cmp32("out_std.npy", &out.std);
    cmp32("out_sem.npy", &out.sem);

    eprintln!("{dataset}: 2D LUT build + apply all fields bit-exact");
}

#[test]
fn nosplit_lut_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        if !is_lut(cfg, "no", 1) {
            continue;
        }
        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        // No-split: HistoBBox1d with delta=None (do_split=False). q -> allow_pos0_neg=false.
        let (built, bin_centers) = build_bbox_lut_1d(
            &vec_f64(&dir, "pos0_center_unscaled.npy"),
            None,
            Some(&mask(&dir)),
            npt,
            false,
            None,
            None,
        );
        validate_lut_1d(&dir, &manifest.dataset, cfg, built, bin_centers);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no no-split-lut 1D golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn bbox_lut_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        if !is_lut(cfg, "bbox", 1) {
            continue;
        }
        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let pos0 = vec_f64(&dir, "pos0_center_unscaled.npy");
        let dpos0 = vec_f64(&dir, "pos0_delta.npy");
        let (built, bin_centers) = build_bbox_lut_1d(
            &pos0,
            Some(&dpos0),
            Some(&mask(&dir)),
            npt,
            false,
            None,
            None,
        );
        validate_lut_1d(&dir, &manifest.dataset, cfg, built, bin_centers);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no bbox-lut 1D golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn full_lut_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        if !is_lut(cfg, "full", 1) {
            continue;
        }
        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        // 1D full does not forward chiDiscAtPi/pos1_period -> defaults True, 2π.
        let (built, bin_centers) = build_full_lut_1d(
            &corners_f64(&dir),
            Some(&mask(&dir)),
            npt,
            false,
            true,
            2.0 * std::f64::consts::PI,
            None,
            None,
        );
        validate_lut_1d(&dir, &manifest.dataset, cfg, built, bin_centers);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no full-lut 1D golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn nosplit_lut_2d_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        if !is_lut(cfg, "no", 2) {
            continue;
        }
        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi: cfg["chi_disc_at_pi"].as_bool().unwrap_or(true),
            pos1_period: cfg["pos1_period"].as_f64().expect("pos1_period"),
            radial_range: None,
            azimuth_range: None,
        };
        let (built, c0, c1) = build_bbox_lut_2d(
            &vec_f64(&dir, "pos0_center_unscaled.npy"),
            None,
            &vec_f64(&dir, "chi_center.npy"),
            None,
            Some(&mask(&dir)),
            (npt_rad, npt_azim),
            &bounds,
        );
        validate_lut_2d(&dir, &manifest.dataset, cfg, built, c0, c1);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no no-split-lut 2D golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn bbox_lut_2d_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        if !is_lut(cfg, "bbox", 2) {
            continue;
        }
        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi: cfg["chi_disc_at_pi"].as_bool().unwrap_or(true),
            pos1_period: cfg["pos1_period"].as_f64().expect("pos1_period"),
            radial_range: None,
            azimuth_range: None,
        };
        let (built, c0, c1) = build_bbox_lut_2d(
            &vec_f64(&dir, "pos0_center_unscaled.npy"),
            Some(&vec_f64(&dir, "pos0_delta.npy")),
            &vec_f64(&dir, "chi_center.npy"),
            Some(&vec_f64(&dir, "chi_delta.npy")),
            Some(&mask(&dir)),
            (npt_rad, npt_azim),
            &bounds,
        );
        validate_lut_2d(&dir, &manifest.dataset, cfg, built, c0, c1);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no bbox-lut 2D golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn full_lut_2d_build_and_apply_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;
        if !is_lut(cfg, "full", 2) {
            continue;
        }
        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        // 2D full forwards chiDiscAtPi and pos1_period = unit1.period (360).
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi: cfg["chi_disc_at_pi"].as_bool().unwrap_or(true),
            pos1_period: cfg["pos1_period"].as_f64().expect("pos1_period"),
            radial_range: None,
            azimuth_range: None,
        };
        let (built, c0, c1) = build_full_lut_2d(
            &corners_f64(&dir),
            Some(&mask(&dir)),
            (npt_rad, npt_azim),
            &bounds,
        );
        validate_lut_2d(&dir, &manifest.dataset, cfg, built, c0, c1);
        checked += 1;
    }
    assert!(
        checked > 0,
        "no full-lut 2D golden datasets found; run golden/gen_golden.py"
    );
}
