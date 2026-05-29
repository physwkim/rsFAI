//! Fiber/GI integrator gate (M10.1): `FiberIntegrator::integrate2d_fiber` /
//! `integrate_fiber` vs pyFAI's `FiberIntegrator` on `golden/datasets_fiber_integrator/`.
//!
//! `gen_golden_fiber_integrator.py` builds a Pilatus1M `FiberIntegrator`, a
//! deterministic image, and the 2D fiber map + both 1D folds for a matrix of GI
//! params and fiber-unit pairs (method `("no","histogram","cython")`, no error
//! model). Here we rebuild the geometry from the committed `.poni`, run the
//! same calls, and compare the populated fields (pyFAI returns no variance for
//! fiber).
//!
//! The fiber position arrays (qip/qoop) carry the scipy-quaternion-vs-direct-matrix
//! ULP divergence (`datasets_fiber` gate); the worry is whether that flips any
//! pixel's histogram bin. This test prints the per-field report so the actual
//! behaviour is measured, then gates: bit-exact where it holds, else a recorded
//! relative tolerance for the fields a flipped pixel perturbs.

use std::path::PathBuf;

use rsfai::{AzimuthalIntegrator, Corrections, IntegrationOptions};
use rsfai_core::compare::{compare_f32, compare_f64};
use rsfai_core::golden::{load_image_f32, load_npy_f32, load_npy_f64};
use rsfai_fiber::{FiberAxes, FiberIntegrator, FiberUnit};
use rsfai_geometry::GiParams;

/// One golden combo, mirroring the generator's `COMBOS`.
struct Combo {
    tag: &'static str,
    incident_angle: f64,
    tilt_angle: f64,
    sample_orientation: u8,
    unit_ip: FiberUnit,
    unit_oop: FiberUnit,
    npt_ip: usize,
    npt_oop: usize,
    correct_solid_angle: bool,
}

const COMBOS: [Combo; 5] = [
    Combo {
        tag: "qnm_so1_sa",
        incident_angle: 0.2,
        tilt_angle: 0.0,
        sample_orientation: 1,
        unit_ip: FiberUnit::QIP_NM,
        unit_oop: FiberUnit::QOOP_NM,
        npt_ip: 200,
        npt_oop: 200,
        correct_solid_angle: true,
    },
    Combo {
        tag: "qnm_tilt_so1",
        incident_angle: 0.2,
        tilt_angle: 0.1,
        sample_orientation: 1,
        unit_ip: FiberUnit::QIP_NM,
        unit_oop: FiberUnit::QOOP_NM,
        npt_ip: 150,
        npt_oop: 180,
        correct_solid_angle: true,
    },
    Combo {
        tag: "qA_so2",
        incident_angle: 0.15,
        tilt_angle: -0.05,
        sample_orientation: 2,
        unit_ip: FiberUnit::QIP_A,
        unit_oop: FiberUnit::QOOP_A,
        npt_ip: 200,
        npt_oop: 200,
        correct_solid_angle: true,
    },
    Combo {
        tag: "exit_deg_so1",
        incident_angle: 0.2,
        tilt_angle: 0.0,
        sample_orientation: 1,
        unit_ip: FiberUnit::EXIT_ANGLE_HORZ_DEG,
        unit_oop: FiberUnit::EXIT_ANGLE_VERT_DEG,
        npt_ip: 120,
        npt_oop: 120,
        correct_solid_angle: true,
    },
    Combo {
        tag: "qnm_nosa",
        incident_angle: 0.2,
        tilt_angle: 0.1,
        sample_orientation: 1,
        unit_ip: FiberUnit::QIP_NM,
        unit_oop: FiberUnit::QOOP_NM,
        npt_ip: 200,
        npt_oop: 200,
        correct_solid_angle: false,
    },
];

/// Relative tolerance for the position-derived fields (`Gate::Tol`): the
/// bin-center axes (a linspace over the ULP-divergent qip/qoop min/max) and the
/// 1D folds (numpy's SIMD-order pairwise reduction, which the scalar
/// `pairwise_sum` port cannot match bit-for-bit). The fiber position arrays
/// diverge from pyFAI by ≤ ~5e-13 relative (scipy-quaternion vs direct matrix);
/// the measured worst across all such fields here is 5.08e-14. 1e-11 is ~200×
/// over that — tight enough to catch a real regression, loose enough to absorb
/// the sanctioned position-ULP and reduction-order noise.
const FIBER_INT_REL_BUDGET: f64 = 1e-11;

/// Which gate a field must clear. The 2D histogram accumulators are pure
/// engine output, bit-identical to pyFAI's serial cython, so they are gated
/// [`Gate::Exact`]: a future libm change that flips a pixel's bin assignment
/// (legal under the Tier-B position-ULP budget) fails here on purpose, flagging
/// it for a re-baseline rather than passing silently. Everything derived from
/// the divergent position arrays is gated [`Gate::Tol`].
#[derive(Clone, Copy)]
enum Gate {
    Exact,
    Tol,
}

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_fiber_integrator")
}

fn f64v(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

fn f32v(name: &str) -> Vec<f32> {
    let p = datasets_root().join(name);
    load_npy_f32(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("C-contiguous")
        .to_vec()
}

/// `true` when the comparison clears its gate (bit-exact always clears; the
/// relative budget only counts for [`Gate::Tol`]).
fn passes(bit_exact: bool, within: bool, gate: Gate) -> bool {
    match gate {
        Gate::Exact => bit_exact,
        Gate::Tol => bit_exact || within,
    }
}

/// Report a single f64 field against its gate, update the running flags.
fn check_f64(
    label: &str,
    got: &[f64],
    golden: &[f64],
    gate: Gate,
    worst_rel: &mut f64,
    fails: &mut usize,
) {
    let r = compare_f64(got, golden);
    if r.max_rel_diff > *worst_rel {
        *worst_rel = r.max_rel_diff;
    }
    let ok = passes(r.is_bit_exact(), r.within_rel(FIBER_INT_REL_BUDGET), gate);
    if !ok {
        *fails += 1;
    }
    eprintln!(
        "    {label:28} {}  ulp={:6} rel={:e} mism={}/{}",
        status(r.is_bit_exact(), ok),
        r.max_ulp,
        r.max_rel_diff,
        r.bit_mismatches,
        r.total
    );
}

fn check_f32(label: &str, got: &[f32], golden: &[f32], gate: Gate, fails: &mut usize) {
    let r = compare_f32(got, golden);
    let ok = passes(r.is_bit_exact(), r.within_rel(FIBER_INT_REL_BUDGET), gate);
    if !ok {
        *fails += 1;
    }
    eprintln!(
        "    {label:28} {}  ulp={:6} rel={:e} mism={}/{}",
        status(r.is_bit_exact(), ok),
        r.max_ulp,
        r.max_rel_diff,
        r.bit_mismatches,
        r.total
    );
}

fn status(bit_exact: bool, ok: bool) -> &'static str {
    if bit_exact {
        "BIT-EXACT"
    } else if ok {
        "rel-ok   "
    } else {
        "FAIL     "
    }
}

#[test]
fn fiber_integrator_matches_pyfai_golden() {
    let root = datasets_root();
    let ai = AzimuthalIntegrator::load(root.join("geometry.poni")).expect("load poni");
    let image = load_image_f32(root.join("data.npy")).expect("load data");

    let mut worst_rel = 0f64;
    let mut fails = 0usize;

    for c in &COMBOS {
        let tag = c.tag;
        let gi = GiParams {
            incident_angle: c.incident_angle,
            tilt_angle: c.tilt_angle,
            sample_orientation: c.sample_orientation,
        };
        let fi = FiberIntegrator::new(ai.clone(), gi);
        let axes = FiberAxes {
            npt_ip: c.npt_ip,
            unit_ip: c.unit_ip,
            ip_range: None,
            npt_oop: c.npt_oop,
            unit_oop: c.unit_oop,
            oop_range: None,
        };
        let opts = IntegrationOptions {
            correct_solid_angle: c.correct_solid_angle,
            ..Default::default()
        };
        let corr = Corrections::with_normalization(1.0);

        eprintln!(
            "=== {tag}  inc={} tilt={} so={} csa={} ===",
            c.incident_angle, c.tilt_angle, c.sample_orientation, c.correct_solid_angle
        );
        let r2 = fi.integrate2d_fiber(&image, &axes, &opts, &corr);
        // Bin-center axes: a linspace over the ULP-divergent qip/qoop min/max.
        check_f64(
            &format!("{tag}/2d/inplane"),
            &r2.inplane,
            &f64v(&format!("{tag}__2d__inplane.npy")),
            Gate::Tol,
            &mut worst_rel,
            &mut fails,
        );
        check_f64(
            &format!("{tag}/2d/outofplane"),
            &r2.outofplane,
            &f64v(&format!("{tag}__2d__outofplane.npy")),
            Gate::Tol,
            &mut worst_rel,
            &mut fails,
        );
        // Histogram accumulators: pure engine output, bit-identical to cython.
        check_f64(
            &format!("{tag}/2d/sum_signal"),
            &r2.sum_signal,
            &f64v(&format!("{tag}__2d__sum_signal.npy")),
            Gate::Exact,
            &mut worst_rel,
            &mut fails,
        );
        check_f64(
            &format!("{tag}/2d/sum_norm"),
            &r2.sum_normalization,
            &f64v(&format!("{tag}__2d__sum_normalization.npy")),
            Gate::Exact,
            &mut worst_rel,
            &mut fails,
        );
        check_f64(
            &format!("{tag}/2d/count"),
            &r2.count,
            &f64v(&format!("{tag}__2d__count.npy")),
            Gate::Exact,
            &mut worst_rel,
            &mut fails,
        );
        check_f32(
            &format!("{tag}/2d/intensity"),
            &r2.intensity,
            &f32v(&format!("{tag}__2d__intensity.npy")),
            Gate::Exact,
            &mut fails,
        );

        for (vert, vtag) in [(true, "v"), (false, "h")] {
            let r1 = fi.integrate_fiber(&image, &axes, vert, &opts, &corr);
            let pre = format!("{tag}__1d{vtag}");
            // The 1D axis is the surviving bin-center axis (Tol); the folded
            // accumulators carry numpy's SIMD-order reduction noise (Tol).
            check_f64(
                &format!("{tag}/1d{vtag}/axis"),
                &r1.axis,
                &f64v(&format!("{pre}__integrated.npy")),
                Gate::Tol,
                &mut worst_rel,
                &mut fails,
            );
            check_f64(
                &format!("{tag}/1d{vtag}/intensity"),
                &r1.intensity,
                &f64v(&format!("{pre}__intensity.npy")),
                Gate::Tol,
                &mut worst_rel,
                &mut fails,
            );
            check_f64(
                &format!("{tag}/1d{vtag}/sum_signal"),
                &r1.sum_signal,
                &f64v(&format!("{pre}__sum_signal.npy")),
                Gate::Tol,
                &mut worst_rel,
                &mut fails,
            );
            check_f64(
                &format!("{tag}/1d{vtag}/sum_norm"),
                &r1.sum_normalization,
                &f64v(&format!("{pre}__sum_normalization.npy")),
                Gate::Tol,
                &mut worst_rel,
                &mut fails,
            );
            check_f64(
                &format!("{tag}/1d{vtag}/count"),
                &r1.count,
                &f64v(&format!("{pre}__count.npy")),
                Gate::Tol,
                &mut worst_rel,
                &mut fails,
            );
        }
    }

    eprintln!("\nworst rel across f64 fields: {worst_rel:e} (budget {FIBER_INT_REL_BUDGET:e})");
    assert_eq!(
        fails, 0,
        "{fails} fiber-integrator field(s) failed their gate"
    );
}
