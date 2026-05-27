//! Validation of the direct-split bbox histogram engines [`histogram1d_bbox`] /
//! [`histogram2d_bbox`] vs pyFAI's golden `Integrate1dtpl`/`Integrate2dtpl`
//! fields (`splitBBox.histoBBox1d_engine` / `histoBBox2d_engine`,
//! `("bbox", "histogram", "cython")`), fed the identical per-pixel arrays pyFAI
//! binned: the unscaled radial center (`pos0_center_unscaled.npy`), the radial
//! half-width (`pos0_delta.npy`), the radian azimuthal center / half-width
//! (`chi_center.npy` / `chi_delta.npy`, 2D only), and the preproc rows
//! (`preproc.npy`).
//!
//! The engines scatter each pixel's split into bins **serially** in pixel-index
//! order, reproducing pyFAI's single-threaded accumulation bit-for-bit, so every
//! field — the bin-center axes AND the accumulator-derived sums (signal/variance/
//! normalization/count/norm²/intensity/std/sem/sigma) — is validated **bit-exact**
//! against the golden. See `doc/bit-exact-ladder.md`.
//!
//! The 1D engine exposes the binned sums at full f64 (the `CsrIntegrate1d`
//! container, matching the CSR / 2D-histogram path — NOT the f32 no-split 1D
//! histogram). Both engines bin the unscaled radial / radian azimuthal and the
//! integrator reports `axis * unit.scale`; for `q_nm^-1` the radial scale is
//! exactly 1.0.

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_integrate::{histogram1d_bbox, histogram2d_bbox, Bbox2dBounds};

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
