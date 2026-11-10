//! Dioptas parity gate for `ring_search` on `golden/datasets_ringsearch/`.
//!
//! `gen_golden_ringsearch.py` runs Dioptas 0.13.0 (pyFAI 2026.9.0) headless
//! on a synthetic LaB6 image and records every ring of its automatic
//! refinement. Gates:
//!
//!   * calibrant 2θ at the wavelength: **bit-exact**.
//!   * [`tth_array`] vs `twoThetaArray`: within [`TTH_TOL`]. pyFAI evaluates
//!     the pixel positions in float32 there (`fromfunction(..., float32)`);
//!     the port stays in f64.
//!   * [`ring_band`] from the geometry each ring was searched with: the area
//!     **exact** but for pixels within [`TTH_TOL`] of the band edge, `keep`
//!     exact, `Imin` within [`IMIN_REL_TOL`] (numpy sums pairwise).
//!   * `Massif` given Dioptas's area, cut and visit order: the peaks
//!     **bit-exact**.
//!   * [`auto_refine`] end to end: its own visit order picks other peaks and
//!     Nelder-Mead is not SLSQP, so the refined geometry is held to being
//!     as good as Dioptas's, within [`AS_GOOD`]: in the largest 2θ gap to
//!     the truth over the image (poni and tilt trade off on a small 2θ
//!     range, the 2θ map does not), and in chi2 on Dioptas's own peaks.

use std::path::PathBuf;

use rsfai::calib::{
    auto_refine, ring_band, tth_array, AutoRefine, GeometryParams, GeometryRefinement, RingSearch,
};
use rsfai::calibrant::Calibrant;
use rsfai::detectors::Detector;
use rsfai::golden::{load_npy_f32, load_npy_f64, load_npy_i32};
use rsfai::peaks::Massif;
use serde_json::Value;

/// float32 pixel positions in pyFAI's `twoThetaArray` (measured 2.3e-8 rad).
const TTH_TOL: f64 = 5e-8;
const IMIN_REL_TOL: f64 = 1e-12;
/// How much worse than Dioptas's the refined geometry may fit (measured
/// 1.3x in the 2θ gap, 1.1x in chi2).
const AS_GOOD: f64 = 2.0;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_ringsearch")
}

fn manifest() -> Value {
    let p = root().join("manifest.json");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read manifest {p:?}: {e}"));
    serde_json::from_str(&t).expect("parse manifest")
}

fn f64v(name: &str) -> Vec<f64> {
    load_npy_f64(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn f32v(name: &str) -> Vec<f32> {
    load_npy_f32(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn i32v(name: &str) -> Vec<i32> {
    load_npy_i32(root().join(name))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .iter()
        .copied()
        .collect()
}

fn param(name: &str) -> GeometryParams {
    let p = f64v(name);
    GeometryParams {
        dist: p[0],
        poni1: p[1],
        poni2: p[2],
        rot1: p[3],
        rot2: p[4],
        rot3: p[5],
        wavelength: p[6],
    }
}

struct Setup {
    image: Vec<f32>,
    shape: (usize, usize),
    detector: Detector,
    calibrant: Calibrant,
    search: RingSearch,
    rings: Vec<usize>,
}

fn setup() -> Setup {
    let m = manifest();
    let c = &m["config"];
    let shape = (
        c["shape"][0].as_u64().unwrap() as usize,
        c["shape"][1].as_u64().unwrap() as usize,
    );
    let pixel = c["pixel"].as_f64().unwrap();
    let mut detector = Detector::generic(pixel, pixel, shape);
    detector.orientation = c["orientation"].as_i64().unwrap() as i32;
    let rings = m["rings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["ring"].as_u64().unwrap() as usize)
        .collect();
    Setup {
        image: f32v("image.npy"),
        shape,
        detector,
        calibrant: Calibrant::from_dspacing(f64v("calibrant_dspacing.npy")),
        search: RingSearch {
            delta_tth_deg: c["delta_tth_deg"].as_f64().unwrap(),
            min_mean_factor: c["min_mean_factor"].as_f64().unwrap(),
            upper_limit: c["upper_limit"].as_f64().unwrap(),
        },
        rings,
    }
}

fn ring_2th(s: &Setup, wavelength: f64) -> Vec<f64> {
    let mut c = s.calibrant.clone();
    c.set_wavelength(wavelength);
    c.get_2th().to_vec()
}

fn max_gap(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn calibrant_2th_is_bit_exact() {
    let s = setup();
    let ours = ring_2th(&s, param("start_param.npy").wavelength);
    let theirs = f64v("calibrant_2th.npy");
    assert_eq!(ours.len(), theirs.len());
    for (a, b) in ours.iter().zip(&theirs) {
        assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
    }
}

#[test]
fn tth_array_matches_two_theta_array() {
    let s = setup();
    let ours = tth_array(&s.detector, &param("start_param.npy"));
    let gap = max_gap(&ours, &f64v("tth_start.npy"));
    println!("tth_array vs twoThetaArray: max gap {gap:.3e} rad");
    assert!(gap < TTH_TOL, "{gap:e}");
}

#[test]
fn ring_band_matches_dioptas() {
    let s = setup();
    let delta = s.search.delta_tth_deg / 180.0 * std::f64::consts::PI;
    for &ring in &s.rings {
        let geo = param(&format!("ring{ring}_param.npy"));
        let tth_ring = ring_2th(&s, geo.wavelength)[ring];
        let tth = tth_array(&s.detector, &geo);
        let band = ring_band(&s.image, &tth, tth_ring, None, &s.search);
        let mut theirs = vec![false; s.image.len()];
        for i in i32v(&format!("ring{ring}_area.npy")) {
            theirs[i as usize] = true;
        }
        let differ: Vec<usize> = (0..s.image.len())
            .filter(|&i| band.area[i] != theirs[i])
            .collect();
        for &i in &differ {
            let edge = ((tth[i] - tth_ring).abs() - delta).abs();
            assert!(
                edge < TTH_TOL,
                "ring {ring}: pixel {i} differs {edge:e} rad off the band edge"
            );
        }
        let cut = f64v(&format!("ring{ring}_cut.npy"));
        if differ.is_empty() {
            assert_eq!(band.keep, cut[1] as usize, "ring {ring} keep");
        }
        if band.keep > 0 {
            let rel = ((band.imin - cut[0]) / cut[0]).abs();
            println!(
                "ring {ring}: imin rel gap {rel:.2e}, {} edge pixels differ",
                differ.len()
            );
            assert!(
                rel < IMIN_REL_TOL,
                "ring {ring} imin {} vs {}",
                band.imin,
                cut[0]
            );
        }
    }
}

#[test]
fn massif_is_bit_exact_given_dioptas_order() {
    let s = setup();
    let (h, w) = s.shape;
    let massif = Massif::new(&s.image, h, w, None);
    for &ring in &s.rings {
        let mut area = vec![false; s.image.len()];
        for i in i32v(&format!("ring{ring}_area.npy")) {
            area[i as usize] = true;
        }
        let order: Vec<(usize, usize)> = i32v(&format!("ring{ring}_order.npy"))
            .into_iter()
            .map(|i| (i as usize / w, i as usize % w))
            .collect();
        let cut = f64v(&format!("ring{ring}_cut.npy"));
        let ours = massif
            .peaks_from_points(&order, &area, cut[0], cut[1] as usize, 0.0)
            .expect("no peak off the image");
        let theirs = f64v(&format!("ring{ring}_peaks.npy"));
        assert_eq!(ours.len() * 2, theirs.len(), "ring {ring} peak count");
        for (k, &(y, x)) in ours.iter().enumerate() {
            assert_eq!(
                y.to_bits(),
                theirs[2 * k].to_bits(),
                "ring {ring} peak {k} y"
            );
            assert_eq!(
                x.to_bits(),
                theirs[2 * k + 1].to_bits(),
                "ring {ring} peak {k} x"
            );
        }
    }
}

#[test]
fn auto_refine_predicts_the_2th_dioptas_does() {
    let s = setup();
    let m = manifest();
    let how = AutoRefine {
        search: s.search,
        rings: m["config"]["num_rings"].as_u64().unwrap() as usize,
        fixed: Vec::new(),
        seed: 1,
    };
    let start = param("start_param.npy");
    let cal = auto_refine(&s.image, None, &s.calibrant, &s.detector, start, &how);
    let ours = tth_array(&s.detector, &cal.param);
    let dioptas = tth_array(&s.detector, &param("final_param.npy"));
    let truth = tth_array(&s.detector, &param("truth_param.npy"));
    let to_dioptas = max_gap(&ours, &dioptas);
    let to_truth = max_gap(&ours, &truth);
    let dioptas_to_truth = max_gap(&dioptas, &truth);
    // Dioptas's own peaks, under our geometry and under its.
    let points: Vec<(f64, f64, usize)> = f64v("final_points.npy")
        .chunks(3)
        .map(|r| (r[0], r[1], r[2] as usize))
        .collect();
    let chi2_ours = GeometryRefinement::new(
        points.clone(),
        s.calibrant.clone(),
        s.detector.clone(),
        cal.param,
    )
    .chi2_current();
    let chi2_dioptas = m["final_chi2"].as_f64().unwrap();
    println!(
        "{} peaks, chi2 {:.3e}; 2θ gap to Dioptas {to_dioptas:.2e}, to truth {to_truth:.2e} \
         (Dioptas to truth {dioptas_to_truth:.2e}) rad; Dioptas's peaks: chi2 {chi2_ours:.3e} ours, \
         {chi2_dioptas:.3e} Dioptas's",
        cal.points.len(),
        cal.chi2
    );
    assert!(!cal.points.is_empty());
    assert!(to_truth <= AS_GOOD * dioptas_to_truth, "{to_truth:e}");
    assert!(chi2_ours <= AS_GOOD * chi2_dioptas, "{chi2_ours:e}");
}
