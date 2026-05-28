//! Validation of the direct-split histogram engines vs pyFAI's golden
//! `Integrate1dtpl`/`Integrate2dtpl` fields:
//! - bbox: [`histogram1d_bbox`] / [`histogram2d_bbox`]
//!   (`splitBBox.histoBBox1d_engine` / `histoBBox2d_engine`,
//!   `("bbox", "histogram", "cython")`), fed the unscaled radial center
//!   (`pos0_center_unscaled.npy`), the radial half-width (`pos0_delta.npy`), the
//!   radian azimuthal center / half-width (`chi_center.npy` / `chi_delta.npy`,
//!   2D only), and the preproc rows (`preproc.npy`).
//! - full: [`histogram1d_full`] / [`histogram2d_full`]
//!   (`splitPixel.fullSplit1D_engine` / `fullSplit2D_engine`,
//!   `("full", "histogram", "cython")`), fed the per-pixel corner array
//!   (`corners.npy`, f32 upcast to f64 — the bits pyFAI's `FullSplitIntegrator`
//!   received) and the preproc rows.
//! - pseudo: [`histogram2d_pseudo`] (`splitPixel.pseudoSplit2D_engine`,
//!   `("pseudo", "histogram", "cython")`, 2D only), fed the same corner array and
//!   preproc rows as full. The engine forwards no `pos1_period`, so the boundary
//!   fold uses `calc_boundaries` with `clip_pos1=False`.
//!
//! Every engine scatters each pixel's split into bins **serially** in pixel-index
//! order, reproducing pyFAI's single-threaded accumulation bit-for-bit, so every
//! field — the bin-center axes AND the accumulator-derived sums (signal/variance/
//! normalization/count/norm²/intensity/std/sem/sigma) — is validated **bit-exact**
//! against the golden. See `doc/bit-exact-ladder.md`.
//!
//! The 1D engines expose the binned sums at full f64 (the `CsrIntegrate1d`
//! container, matching the CSR / 2D-histogram path — NOT the f32 no-split 1D
//! histogram). They bin the unscaled radial / radian azimuthal and the
//! integrator reports `axis * unit.scale`; for `q_nm^-1` the radial scale is
//! exactly 1.0.

use std::f64::consts::PI;
use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_integrate::{
    histogram1d_bbox, histogram1d_full, histogram2d_bbox, histogram2d_full, histogram2d_pseudo,
    Bbox2dBounds,
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
    ErrorModel::from_code(code as i32).unwrap_or_else(|| panic!("unknown error_model_code {code}"))
}

fn load_vec_f64(p: PathBuf) -> Vec<f64> {
    load_npy_f64(p)
        .expect("f64 npy")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn load_vec_f32(p: PathBuf) -> Vec<f32> {
    load_npy_f32(p)
        .expect("f32 npy")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

/// The `(npix, 4, 2)` corner array, flattened C-order and upcast f32 -> f64 — the
/// exact bits pyFAI's `FullSplitIntegrator` received (stored f32, used as
/// position_t f64), matching `csr_golden`'s full-split loader.
fn load_corners_f64(p: PathBuf) -> Vec<f64> {
    load_npy_f32(p)
        .expect("corners f32 npy")
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|&v| v as f64)
        .collect()
}

#[test]
fn histogram1d_bbox_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // Direct-split bbox histogram: method == ("bbox", "histogram", _), dim 1.
        let is_bbox_histogram = cfg["method"]
            .as_array()
            .map(|m| {
                m.len() >= 3 && m[0].as_str() == Some("bbox") && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if cfg["dim"].as_u64().unwrap_or(1) != 1 || !is_bbox_histogram {
            continue;
        }
        // out.position is unscaled; the comparison multiplies by unit_scale, so a
        // non-1.0 scale is handled, but the radial unit is asserted positive (q).
        let unit = cfg["unit"].as_str().unwrap_or("");
        assert_eq!(
            unit, "q_nm^-1",
            "{}: radial check assumes a positive unit (q); record allow_pos0_neg for {unit}",
            manifest.dataset
        );
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");

        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));

        // Engine inputs: the unscaled radial center / half-width and the preproc.
        let radial = load_vec_f64(dir.join("pos0_center_unscaled.npy"));
        let delta0 = load_vec_f64(dir.join("pos0_delta.npy"));
        let prep = load_vec_f32(dir.join("preproc.npy"));
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // q is positive -> allow_pos0_neg = false. empty = 0.0 (fresh integrator).
        let out = histogram1d_bbox(
            &radial,
            &delta0,
            &prep,
            Some(&mask),
            npt,
            error_model,
            0.0,
            false,
        );

        // Radial bin centers (f64) -> bit-exact after scaling.
        let scaled_rad: Vec<f64> = out.position.iter().map(|&p| p * unit_scale).collect();
        let g_radial = load_vec_f64(dir.join("out_radial.npy"));
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );

        // f64 binned sums (CsrIntegrate1d) -> bit-exact (serial scatter).
        let check_f64 = |file: &str, actual: &[f64], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_vec_f64(p);
            let r = compare_f64(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };
        // f32 derived fields -> bit-exact.
        let check_f32 = |file: &str, actual: &[f32], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_vec_f32(p);
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };

        check_f64("out_sum_signal.npy", &out.sum_signal, true);
        check_f64("out_sum_normalization.npy", &out.sum_normalization, true);
        check_f64("out_count.npy", &out.count, true);
        check_f32("out_intensity.npy", &out.intensity, true);

        // Error-model-only fields (exposed when do_variance): variance, norm²,
        // and the libc-sqrt std/sem/sigma.
        let mut em = 0;
        em += i32::from(check_f64("out_sum_variance.npy", &out.sum_variance, false));
        em += i32::from(check_f64(
            "out_sum_normalization2.npy",
            &out.sum_norm_sq,
            false,
        ));
        em += i32::from(check_f32("out_std.npy", &out.std, false));
        em += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em > 0,
                "{}: error-model dataset exposed no variance/std/sem fields",
                manifest.dataset
            );
        }
        eprintln!(
            "{}: 1D bbox all fields bit-exact (radial max_ulp={}); {em} error-model fields",
            manifest.dataset, r_radial.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 1D bbox-histogram golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn histogram2d_bbox_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        let is_bbox_histogram = cfg["method"]
            .as_array()
            .map(|m| {
                m.len() >= 3 && m[0].as_str() == Some("bbox") && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if cfg["dim"].as_u64().unwrap_or(1) != 2 || !is_bbox_histogram {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));

        // Engine inputs: the unscaled radial center / half-width and the radian
        // azimuthal center / half-width pyFAI binned.
        let radial = load_vec_f64(dir.join("pos0_center_unscaled.npy"));
        let delta0 = load_vec_f64(dir.join("pos0_delta.npy"));
        let azimuthal = load_vec_f64(dir.join("chi_center.npy"));
        let delta1 = load_vec_f64(dir.join("chi_delta.npy"));
        let prep = load_vec_f32(dir.join("preproc.npy"));
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // Standard radial unit (q) cannot be negative -> allow_pos0_neg = false.
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi,
            pos1_period,
        };

        let out = histogram2d_bbox(
            &radial,
            &delta0,
            &azimuthal,
            &delta1,
            &prep,
            Some(&mask),
            (npt_rad, npt_azim),
            &bounds,
            error_model,
            0.0,
        );

        // Radial / azimuthal bin centers, scaled to the reported axes -> bit-exact.
        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = load_vec_f64(dir.join("out_radial.npy"));
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );
        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = load_vec_f64(dir.join("out_azimuthal.npy"));
        let r_azim = compare_f64(&scaled_azim, &g_azim);
        assert!(
            r_azim.is_bit_exact(),
            "{}: out_azimuthal not bit-exact: {r_azim:?}",
            manifest.dataset
        );

        // Accumulator fields -> bit-exact (serial scatter). The 2D engine exposes
        // binned sums at full f64 (NOT downcast to f32 like the 1D histogram).
        let check_f64 = |file: &str, actual: &[f64], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_vec_f64(p);
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
            let g = load_vec_f32(p);
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

        let mut em = 0;
        em += i32::from(check_f64("out_sum_variance.npy", &out.variance, false));
        em += i32::from(check_f64("out_sum_normalization2.npy", &out.norm_sq, false));
        em += i32::from(check_f32("out_std.npy", &out.std, false));
        em += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em > 0,
                "{}: error-model dataset exposed no variance/std/sem fields",
                manifest.dataset
            );
        }
        eprintln!(
            "{}: 2D bbox all fields bit-exact (radial ulp={}, azim ulp={}); {em} error-model fields",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D bbox-histogram golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn histogram1d_full_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // Full pixel-splitting histogram: method == ("full", "histogram", _), dim 1.
        let is_full_histogram = cfg["method"]
            .as_array()
            .map(|m| {
                m.len() >= 3 && m[0].as_str() == Some("full") && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if cfg["dim"].as_u64().unwrap_or(1) != 1 || !is_full_histogram {
            continue;
        }
        let unit = cfg["unit"].as_str().unwrap_or("");
        assert_eq!(
            unit, "q_nm^-1",
            "{}: radial check assumes a positive unit (q); record allow_pos0_neg for {unit}",
            manifest.dataset
        );
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");

        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));

        // Engine inputs: the per-pixel corner array and the preproc rows.
        let corners = load_corners_f64(dir.join("corners.npy"));
        let prep = load_vec_f32(dir.join("preproc.npy"));
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // q is positive -> allow_pos0_neg = false. The 1D full setup (common.py)
        // does NOT forward chiDiscAtPi / pos1_period to FullSplitIntegrator, so
        // they take the constructor defaults chiDiscAtPi=True, pos1_period=2π.
        let out = histogram1d_full(
            &corners,
            &prep,
            Some(&mask),
            npt,
            error_model,
            0.0,
            false,
            true,
            2.0 * PI,
        );

        let scaled_rad: Vec<f64> = out.position.iter().map(|&p| p * unit_scale).collect();
        let g_radial = load_vec_f64(dir.join("out_radial.npy"));
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );

        let check_f64 = |file: &str, actual: &[f64], required: bool| -> bool {
            let p = dir.join(file);
            if !p.exists() {
                assert!(!required, "{}: missing required {file}", manifest.dataset);
                return false;
            }
            let g = load_vec_f64(p);
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
            let g = load_vec_f32(p);
            let r = compare_f32(actual, &g);
            assert!(
                r.is_bit_exact(),
                "{}: {file} not bit-exact: {r:?}",
                manifest.dataset
            );
            true
        };

        check_f64("out_sum_signal.npy", &out.sum_signal, true);
        check_f64("out_sum_normalization.npy", &out.sum_normalization, true);
        check_f64("out_count.npy", &out.count, true);
        check_f32("out_intensity.npy", &out.intensity, true);

        let mut em = 0;
        em += i32::from(check_f64("out_sum_variance.npy", &out.sum_variance, false));
        em += i32::from(check_f64(
            "out_sum_normalization2.npy",
            &out.sum_norm_sq,
            false,
        ));
        em += i32::from(check_f32("out_std.npy", &out.std, false));
        em += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em > 0,
                "{}: error-model dataset exposed no variance/std/sem fields",
                manifest.dataset
            );
        }
        eprintln!(
            "{}: 1D full all fields bit-exact (radial max_ulp={}); {em} error-model fields",
            manifest.dataset, r_radial.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 1D full-histogram golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn histogram2d_full_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        let is_full_histogram = cfg["method"]
            .as_array()
            .map(|m| {
                m.len() >= 3 && m[0].as_str() == Some("full") && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if cfg["dim"].as_u64().unwrap_or(1) != 2 || !is_full_histogram {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        // 2D full setup (common.py) forwards chiDiscAtPi and pos1_period = unit1.period
        // (360, applied to radian azimuths — a pyFAI quirk); both from the manifest.
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));

        let corners = load_corners_f64(dir.join("corners.npy"));
        let prep = load_vec_f32(dir.join("preproc.npy"));
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // Standard radial unit (q) cannot be negative -> allow_pos0_neg = false.
        let bounds = Bbox2dBounds {
            allow_pos0_neg: false,
            chi_disc_at_pi,
            pos1_period,
        };

        let out = histogram2d_full(
            &corners,
            &prep,
            Some(&mask),
            (npt_rad, npt_azim),
            &bounds,
            error_model,
            0.0,
        );

        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = load_vec_f64(dir.join("out_radial.npy"));
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );
        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = load_vec_f64(dir.join("out_azimuthal.npy"));
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
            let g = load_vec_f64(p);
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
            let g = load_vec_f32(p);
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

        let mut em = 0;
        em += i32::from(check_f64("out_sum_variance.npy", &out.variance, false));
        em += i32::from(check_f64("out_sum_normalization2.npy", &out.norm_sq, false));
        em += i32::from(check_f32("out_std.npy", &out.std, false));
        em += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em > 0,
                "{}: error-model dataset exposed no variance/std/sem fields",
                manifest.dataset
            );
        }
        eprintln!(
            "{}: 2D full all fields bit-exact (radial ulp={}, azim ulp={}); {em} error-model fields",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D full-histogram golden datasets found; run golden/gen_golden.py"
    );
}

#[test]
fn histogram2d_pseudo_bit_exact() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        let is_pseudo_histogram = cfg["method"]
            .as_array()
            .map(|m| {
                m.len() >= 3
                    && m[0].as_str() == Some("pseudo")
                    && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if cfg["dim"].as_u64().unwrap_or(1) != 2 || !is_pseudo_histogram {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        // The pseudo engine forwards no pos1_period (calc_boundaries clip_pos1=False);
        // it passes chiDiscAtPi=self.chiDiscAtPi and allow_pos0_neg=not radial_unit.positive.
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));

        let corners = load_corners_f64(dir.join("corners.npy"));
        let prep = load_vec_f32(dir.join("preproc.npy"));
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let out = histogram2d_pseudo(
            &corners,
            &prep,
            Some(&mask),
            (npt_rad, npt_azim),
            false, // standard radial unit (q) cannot be negative
            chi_disc_at_pi,
            error_model,
            0.0,
        );

        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = load_vec_f64(dir.join("out_radial.npy"));
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );
        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = load_vec_f64(dir.join("out_azimuthal.npy"));
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
            let g = load_vec_f64(p);
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
            let g = load_vec_f32(p);
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

        let mut em = 0;
        em += i32::from(check_f64("out_sum_variance.npy", &out.variance, false));
        em += i32::from(check_f64("out_sum_normalization2.npy", &out.norm_sq, false));
        em += i32::from(check_f32("out_std.npy", &out.std, false));
        em += i32::from(check_f32("out_sem.npy", &out.sem, false));
        em += i32::from(check_f32("out_sigma.npy", &out.sigma, false));
        if error_model != ErrorModel::No {
            assert!(
                em > 0,
                "{}: error-model dataset exposed no variance/std/sem fields",
                manifest.dataset
            );
        }
        eprintln!(
            "{}: 2D pseudo all fields bit-exact (radial ulp={}, azim ulp={}); {em} error-model fields",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D pseudo-histogram golden datasets found; run golden/gen_golden.py"
    );
}
