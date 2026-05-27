//! M4 Tier-A validation: the 1D histogram engine [`histogram1d`] must be
//! **bit-exact** vs pyFAI's golden `Integrate1dtpl` fields, fed the identical
//! inputs pyFAI binned — the dumped radial array (`pos0_center.npy`) and the
//! dumped preprocessed rows (`preproc.npy`).
//!
//! The integrator bins the **unscaled** radial (`center_array(unit,
//! scale=False)`) and reports `position * unit.scale`. For `q_nm^-1` the scale
//! is exactly `1.0`, so `pos0_center.npy` is bit-identical to the engine input
//! and `out_radial == position`. This test only runs the `("no", "histogram",
//! ...)` datasets (the histogram engine), all of which use `q_nm^-1` here; a
//! future non-unit-scale histogram golden would need the scale factor recorded.

use std::path::PathBuf;

use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::dtype::ErrorModel;
use rsfai_core::golden::{load_manifest, load_npy_f32, load_npy_f64};
use rsfai_integrate::histogram1d;

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

#[test]
fn histogram1d_bit_exact() {
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

        // Each remaining field is f32. Mandatory fields are always exposed; the
        // error-model fields (variance/norm²/std/sem/sigma) only when
        // do_variance, so they are checked when the dataset dumped them. Returns
        // whether the file existed (and was therefore validated).
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
            "{}: all fields bit-exact (radial max_ulp={}, {em_fields} error-model fields)",
            manifest.dataset, r_radial.max_ulp,
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no histogram golden datasets found; run golden/gen_golden.py"
    );
}
