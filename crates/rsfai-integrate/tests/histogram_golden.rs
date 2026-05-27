//! Validation of the 1D/2D histogram engines vs pyFAI's golden
//! `Integrate1dtpl`/`Integrate2dtpl` fields, fed the identical inputs pyFAI
//! binned — the dumped radial array (`pos0_center.npy`) and preprocessed rows
//! (`preproc.npy`).
//!
//! The histogram **accumulation** is parallelized with a rayon fold/reduce
//! (non-deterministic f64 add order), so the accumulator-derived fields
//! (signal/variance/normalization/count/norm²/intensity/std/sem/sigma) are
//! **not** bit-exact against the serial golden; they are validated at relative
//! error `<= REL_TOL` (1e-6). The bin-center **axes** (`out_radial` /
//! `out_azimuthal`) derive from the order-independent min/max + `linspace`, so
//! they stay **bit-exact**. See `doc/bit-exact-ladder.md`.
//!
//! The integrator bins the **unscaled** radial (`center_array(unit,
//! scale=False)`) and reports `position * unit.scale`. For `q_nm^-1` the scale
//! is exactly `1.0`, so `pos0_center.npy` is bit-identical to the engine input
//! and `out_radial == position`. The 1D test only runs the `("no", "histogram",
//! ...)` datasets, all of which use `q_nm^-1` here; a future non-unit-scale
//! histogram golden would need the scale factor recorded.

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64, load_npy_i8};
use rsfai_integrate::{histogram1d, histogram2d, Hist2dOptions};

/// Relative-error gate for the parallel-histogram accumulator fields (the f64
/// reorder error is ~n·eps ≈ 2e-10, well under this; see the module docs).
const REL_TOL: f64 = 1e-6;

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

#[test]
fn histogram1d_within_tolerance() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // Only the pure-Cython histogram engine: method == ("no", "histogram", _).
        let method = cfg["method"].as_array();
        let is_histogram = method
            .map(|m| {
                m.len() >= 3 && m[0].as_str() == Some("no") && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if !is_histogram {
            continue;
        }
        // The 2D histogram datasets share the ("no", "histogram") method tuple;
        // skip them here (they carry npt_rad/npt_azim, not npt) — covered by
        // `histogram2d_bit_exact`.
        if cfg["dim"].as_u64().unwrap_or(1) != 1 {
            continue;
        }
        // This test compares `position` to `out_radial` directly, which only
        // holds when unit.scale == 1.0 (q_nm^-1). Guard against silently
        // mis-validating a future scaled unit.
        let unit = cfg["unit"].as_str().unwrap_or("");
        assert_eq!(
            unit, "q_nm^-1",
            "{}: histogram radial check assumes unit.scale == 1.0 (q_nm^-1); \
             record unit.scale to validate {unit}",
            manifest.dataset
        );

        let npt = cfg["npt"].as_u64().expect("npt") as usize;
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        let radial_range = cfg["radial_range"].as_array().map(|r| {
            (
                r[0].as_f64().expect("radial_range[0]"),
                r[1].as_f64().expect("radial_range[1]"),
            )
        });

        // Engine inputs: the radial array pyFAI binned and the preproc rows.
        let radial = load_npy_f64(dir.join("pos0_center.npy"))
            .expect("pos0_center")
            .as_slice()
            .expect("contiguous")
            .to_vec();
        let prep = load_npy_f32(dir.join("preproc.npy"))
            .expect("preproc")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // empty = self._empty, default 0.0 for a freshly loaded integrator.
        let out = histogram1d(&radial, &prep, npt, radial_range, error_model, 0.0);

        // Radial bin centers (f64). For q_nm^-1 (scale 1.0), out_radial == position.
        let g_radial = load_npy_f64(dir.join("out_radial.npy"))
            .expect("out_radial")
            .as_slice()
            .unwrap()
            .to_vec();
        let r_radial = compare_f64(&out.position, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );

        // Each remaining field is f32 and is an accumulator output of the
        // parallel histogram -> validated at relative error <= REL_TOL (not
        // bitwise). Mandatory fields are always exposed; the error-model fields
        // (variance/norm²/std/sem/sigma) only when do_variance, so they are
        // checked when the dataset dumped them. Returns whether the file existed
        // (and was therefore validated).
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
                r.within_rel(REL_TOL),
                "{}: {file} exceeds rel tol {REL_TOL:e}: {r:?}",
                manifest.dataset
            );
            true
        };

        check_f32("out_sum_signal.npy", &out.signal, true);
        check_f32("out_sum_normalization.npy", &out.normalization, true);
        check_f32("out_count.npy", &out.count, true);
        check_f32("out_intensity.npy", &out.intensity, true);

        // Error-model-only fields: norm² (f64-multiply path), variance, and the
        // libc-sqrt std/sem/sigma. pyFAI exposes these only when do_variance.
        let mut em_fields = 0;
        em_fields += i32::from(check_f32("out_sum_variance.npy", &out.variance, false));
        em_fields += i32::from(check_f32("out_sum_normalization2.npy", &out.norm_sq, false));
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
            "{}: radial bit-exact (max_ulp={}); accumulator fields within rel {REL_TOL:e} \
             ({em_fields} error-model fields)",
            manifest.dataset, r_radial.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no histogram golden datasets found; run golden/gen_golden.py"
    );
}

/// Tier-A validation of the 2D histogram engine [`histogram2d`] vs pyFAI's
/// golden `Integrate2dtpl` fields, fed the identical per-pixel arrays pyFAI
/// binned: the unscaled radial (`pos0_center_unscaled.npy`), the azimuthal
/// centers in radians (`chi_center.npy`, bit-identical to `array_from_unit(
/// "center", chi_deg, scale=False)`), and the preproc rows (`preproc.npy`).
///
/// The engine bins unscaled radial / radian azimuthal; the reported axes are
/// `radial * unit.scale` and `azimuthal * azimuth_unit.scale` (CHI_DEG scale =
/// 180/π). The binned sums are exposed at full f64 (unlike the 1D histogram);
/// the variance-family fields (variance, norm², std, sem, sigma) appear only
/// when `do_variance`, so they are validated when the dataset dumped them.
#[test]
fn histogram2d_within_tolerance() {
    let mut checked = 0usize;
    for dir in dataset_dirs() {
        let manifest = load_manifest(dir.join("manifest.json")).expect("manifest");
        let cfg = &manifest.config;

        // 2D pure-Cython histogram engine: dim == 2, method == ("no","histogram",_).
        if cfg["dim"].as_u64().unwrap_or(1) != 2 {
            continue;
        }
        let is_histogram = cfg["method"]
            .as_array()
            .map(|m| {
                m.len() >= 3 && m[0].as_str() == Some("no") && m[1].as_str() == Some("histogram")
            })
            .unwrap_or(false);
        if !is_histogram {
            continue;
        }

        let npt_rad = cfg["npt_rad"].as_u64().expect("npt_rad") as usize;
        let npt_azim = cfg["npt_azim"].as_u64().expect("npt_azim") as usize;
        let unit_scale = cfg["unit_scale"].as_f64().expect("unit_scale");
        let azim_scale = cfg["azim_scale"].as_f64().expect("azim_scale");
        let pos1_period = cfg["pos1_period"].as_f64().expect("pos1_period");
        let chi_disc_at_pi = cfg["chi_disc_at_pi"].as_bool().unwrap_or(true);
        let error_model = error_model_from_code(cfg["error_model_code"].as_i64().unwrap_or(0));
        let radial_range = cfg["radial_range"].as_array().map(|r| {
            (
                r[0].as_f64().expect("radial_range[0]"),
                r[1].as_f64().expect("radial_range[1]"),
            )
        });
        let azimuth_range = cfg["azimuth_range"].as_array().map(|r| {
            (
                r[0].as_f64().expect("azimuth_range[0]"),
                r[1].as_f64().expect("azimuth_range[1]"),
            )
        });

        // Standard radial unit (q) cannot be negative -> allow_radial_neg=false.
        let opts = Hist2dOptions {
            bins: (npt_rad, npt_azim),
            radial_range,
            azimuth_range,
            error_model,
            allow_radial_neg: false,
            chi_disc_at_pi,
            pos1_period,
            empty: 0.0,
        };

        // Engine inputs: the per-pixel arrays pyFAI binned.
        let radial = load_npy_f64(dir.join("pos0_center_unscaled.npy"))
            .expect("pos0_center_unscaled")
            .as_slice()
            .expect("contiguous")
            .to_vec();
        let azimuthal = load_npy_f64(dir.join("chi_center.npy"))
            .expect("chi_center")
            .as_slice()
            .expect("contiguous")
            .to_vec();
        let prep = load_npy_f32(dir.join("preproc.npy"))
            .expect("preproc")
            .as_slice()
            .expect("contiguous")
            .to_vec();
        let mask = load_npy_i8(dir.join("mask.npy"))
            .expect("mask")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        let out = histogram2d(&radial, &azimuthal, &prep, Some(&mask), &opts);

        // Radial / azimuthal bin centers, scaled to the reported axes.
        let scaled_rad: Vec<f64> = out.radial.iter().map(|&p| p * unit_scale).collect();
        let g_radial = load_npy_f64(dir.join("out_radial.npy"))
            .expect("out_radial")
            .as_slice()
            .unwrap()
            .to_vec();
        let r_radial = compare_f64(&scaled_rad, &g_radial);
        assert!(
            r_radial.is_bit_exact(),
            "{}: out_radial not bit-exact: {r_radial:?}",
            manifest.dataset
        );

        let scaled_azim: Vec<f64> = out.azimuthal.iter().map(|&p| p * azim_scale).collect();
        let g_azim = load_npy_f64(dir.join("out_azimuthal.npy"))
            .expect("out_azimuthal")
            .as_slice()
            .unwrap()
            .to_vec();
        let r_azim = compare_f64(&scaled_azim, &g_azim);
        assert!(
            r_azim.is_bit_exact(),
            "{}: out_azimuthal not bit-exact: {r_azim:?}",
            manifest.dataset
        );

        // Accumulator fields of the parallel histogram -> relative error <=
        // REL_TOL (not bitwise). The 2D engine exposes the binned sums at full
        // f64 (`out_data[...,k].T`), NOT downcast to f32 like the 1D histogram.
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
                r.within_rel(REL_TOL),
                "{}: {file} exceeds rel tol {REL_TOL:e}: {r:?}",
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
                r.within_rel(REL_TOL),
                "{}: {file} exceeds rel tol {REL_TOL:e}: {r:?}",
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
            "{}: 2D axes bit-exact (radial ulp={}, azim ulp={}); accumulator fields within \
             rel {REL_TOL:e} ({em_fields} error-model fields)",
            manifest.dataset, r_radial.max_ulp, r_azim.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no 2D histogram golden datasets found; run golden/gen_golden.py"
    );
}
