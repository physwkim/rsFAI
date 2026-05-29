//! Fiber / grazing-incidence unit-equation gate (M10.0), pure identical-input.
//!
//! `golden/gen_golden_fiber.py` dumps a strided sample of ~4096 Pilatus1M pixels'
//! lab coords `(x, y, z)` plus pyFAI's per-unit center value for each
//! `(incident_angle, tilt_angle, sample_orientation)` combo. Here we feed the
//! SAME `(x, y, z)` to [`fiber_center_array`] and compare.
//!
//! The gate is split by the divergence source, which is measured (not assumed):
//!
//!   * **Bit-exact (0 ULP).** Units that never touch the sample rotation
//!     (`qtot`, `scatvert`, `scathorz` — pure `atan2`/`sqrt` on the remapped
//!     coords) in every combo, AND *all* units in the no-GI combo (a zero-angle
//!     rotation is identity-exact on both sides). A nonzero ULP here is a real
//!     algebra regression.
//!   * **Relative budget ([`FIBER_REL_BUDGET`]).** The rotation-routed units
//!     under a real GI rotation. pyFAI evaluates `unit.equation` via scipy
//!     `Rotation` (a quaternion); rsFAI uses a direct cos/sin matrix, so the two
//!     diverge ≤1 ULP in the matrix, amplified by the q-component cancellations
//!     (`qbeam = k(z/norm−1)` near the beam center, `qhorz` sign change). ULP
//!     near those zero crossings is meaningless; the *relative* error is the
//!     stable bound. Measured worst across all GI arrays: `max_rel ≈ 5.23e-13`
//!     (c2/exithorz). See `fiber_units` docs / kodex `f3389aef`.

use std::path::PathBuf;

use rsfai_core::compare::compare_f64;
use rsfai_core::golden::load_npy_f64;
use rsfai_geometry::{fiber_center_array, FiberSpace, GiParams};

/// Relative-error ceiling for the rotation-routed units under a GI rotation.
/// Measured worst across every GI array is `5.23e-13`; this leaves ~20× margin
/// for libm drift while still catching any rotation-convention regression
/// (which lands at `rel ~ 1e-1`, ten-plus orders away).
const FIBER_REL_BUDGET: f64 = 1e-11;

/// pyFAI wavelength used by `gen_golden_fiber.py` (m).
const WAVELENGTH: f64 = 1e-10;

/// `(incident_angle, tilt_angle, sample_orientation)`, mirroring the generator.
const COMBOS: [(f64, f64, u8); 5] = [
    (0.0, 0.0, 1),
    (0.2, 0.4, 1),
    (0.2, 0.4, 2),
    (0.2, 0.4, 5),
    (0.1, -0.3, 7),
];

/// Generator key → ([`FiberSpace`], routes through the sample rotation).
/// The `rotates` flag mirrors `fiber_equation`: only the q-sample and
/// exit-angle units apply `rotate_q_lab` / `rotate_cartesian`; `qtot` and the
/// scattering angles are pure `atan2`/`sqrt` on the remapped coords.
const UNITS: [(&str, FiberSpace, bool); 10] = [
    ("qip", FiberSpace::Qip, true),
    ("qoop", FiberSpace::Qoop, true),
    ("qtot", FiberSpace::Qtotal, false),
    ("chigi", FiberSpace::ChiGi, true),
    ("qbeam", FiberSpace::Qbeam, true),
    ("qhorz", FiberSpace::Qhorz, true),
    ("scatvert", FiberSpace::ScatteringAngleVert, false),
    ("scathorz", FiberSpace::ScatteringAngleHorz, false),
    ("exitvert", FiberSpace::ExitAngleVert, true),
    ("exithorz", FiberSpace::ExitAngleHorz, true),
];

fn datasets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden/datasets_fiber")
}

fn load_f64(name: &str) -> Vec<f64> {
    let p = datasets_root().join(name);
    load_npy_f64(&p)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .as_slice()
        .expect("golden C-contiguous")
        .to_vec()
}

#[test]
fn fiber_units_match_pyfai_golden() {
    let root = datasets_root();
    assert!(
        root.join("sample_x.npy").exists(),
        "fiber golden missing; run golden/gen_golden_fiber.py"
    );
    let x = load_f64("sample_x.npy");
    let y = load_f64("sample_y.npy");
    let z = load_f64("sample_z.npy");

    let mut worst_rel = 0f64;
    let mut worst_rel_label = String::new();
    let mut fails = 0usize;

    for (ci, &(inc, tilt, so)) in COMBOS.iter().enumerate() {
        let gi = GiParams {
            incident_angle: inc,
            tilt_angle: tilt,
            sample_orientation: so,
        };
        let combo_has_gi = inc != 0.0 || tilt != 0.0;
        eprintln!("=== c{ci}  inc={inc} tilt={tilt} so={so} ===");
        for (key, space, rotates) in UNITS {
            let golden = load_f64(&format!("out_c{ci}__{key}.npy"));
            let got = fiber_center_array(space, 1.0, &x, &y, &z, WAVELENGTH, gi);
            let r = compare_f64(&got, &golden);

            // Bit-exact expected unless this unit's rotation is actually engaged.
            let rel_gated = rotates && combo_has_gi;
            let ok = if rel_gated {
                if r.max_rel_diff > worst_rel {
                    worst_rel = r.max_rel_diff;
                    worst_rel_label = format!("c{ci}/{key}");
                }
                r.within_rel(FIBER_REL_BUDGET)
            } else {
                r.is_bit_exact()
            };
            if !ok {
                fails += 1;
            }
            eprintln!(
                "    {key:9} {:9} {}  max_ulp={:5} max_rel={:e} mismatches={}/{}",
                if rel_gated { "rel" } else { "bit-exact" },
                if ok { "PASS" } else { "FAIL" },
                r.max_ulp,
                r.max_rel_diff,
                r.bit_mismatches,
                r.total
            );
        }
    }

    eprintln!(
        "\nworst GI rel: {worst_rel_label} max_rel={worst_rel:e} (budget {FIBER_REL_BUDGET:e})"
    );
    assert_eq!(
        fails, 0,
        "{fails} fiber unit array(s) failed their gate (bit-exact / {FIBER_REL_BUDGET:e} rel) — see report"
    );
}
