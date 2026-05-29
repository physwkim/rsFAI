#!/usr/bin/env python
"""Golden generator for ControlPoints + GeometryRefinement (rsfai-calib).

Run single-thread in the daq env (pyFAI 2026.5.0 built -ffp-contract=off,
scipy 1.17.1, numpy 2.4.3):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_calib.py

TWO parity surfaces, both feeding crates/rsfai-calib/tests/golden_calib.rs:

  * Fixed-parameter residual / chi2 (the BIT-EXACT gate). For a fixed parameter
    vector (the post-`guess_poni` initial estimate) we dump the control points
    `(d1, d2, ring)`, the calibrant d-spacing list + wavelength, the per-ring
    expected 2theta (`calc_2th`), the measured 2theta (`tth`), the residual
    vector (`residu1`), and chi2 (`residu2`). The Rust side recomputes these
    from the SAME params/points/calibrant and must match bit-for-bit (the
    geometry atan2/sin/cos is the only ULP-budgeted part).

  * Converged params / cost (the TOLERANCE gate). We run pyFAI's `refine2`
    (fmin_slsqp, wavelength fixed) and dump the converged 6-parameter vector and
    the converged chi2. The Rust `refine()` (argmin Nelder-Mead) is compared to
    these at a recorded relative tolerance, with cost_rust <= cost_pyfai.

The dataset is pyFAI's own `test_geometry_refinement.py::test_noSpline` LaB6
fixture (51 control points on 5 rings, 1.5e-5 m pixels, 1.54e-10 m). The fixture
arrays are embedded verbatim below so the generator is self-contained.

Provenance (pyFAI/numpy/scipy version, pixel size, wavelength, fixed params) is
in manifest.json. The fixed-param vector is dumped explicitly so the bit-exact
gate is reproducible and the residual contract is pinned.
"""

import json
import os
import platform

import numpy as np
import scipy
import pyFAI
from pyFAI import calibrant as cal
from pyFAI.geometryRefinement import GeometryRefinement

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_calib")

PIXEL1 = 1.5e-5
PIXEL2 = 1.5e-5
WAVELENGTH = 1.54e-10

# --- pyFAI test_geometry_refinement.py::test_noSpline fixture (verbatim) ---
DATA = [
    [1585.9999996029055, 2893.999999119241, 0.5300564938306779],
    [1853.9999932086102, 2873.000000163791, 0.5300564938306779],
    [2163.9999987531855, 2854.9999987738884, 0.5300564938306779],
    [2699.999997791493, 2893.9999985831755, 0.5300564938306779],
    [3186.9999966428777, 3028.9999985930604, 0.5300564938306779],
    [3595.000003953466, 3167.000002296746, 0.5300564938306779],
    [3835.0000007197755, 3300.000000253641, 0.5300564938306779],
    [1252.000002688137, 2984.0000056421914, 0.5300564938306779],
    [576.9999248635229, 3220.0000014469815, 0.5300564938306779],
    [52.99998954676053, 3531.999997531496, 0.5300564938306779],
    [520.9999986245284, 2424.0000005943775, 0.6532767390214775],
    [1108.00000451895, 2239.9999793751085, 0.6532767390214775],
    [2022.0000098770186, 2136.9999921020726, 0.6532767390214775],
    [2436.000002384907, 2137.0000034435734, 0.6532767390214775],
    [2797.9999973906524, 2169.9999849019205, 0.6532767390214775],
    [3516.0000041508365, 2354.0000059814265, 0.6532767390214775],
    [3870.9999995625412, 2464.9999964079757, 0.6532767390214775],
    [3735.9999952703465, 2417.999988822315, 0.6532767390214775],
    [3374.000142868041, 2289.9999885080188, 0.6532767390214775],
    [1709.99999872134, 2165.000000669327, 0.6532767390214775],
    [2004.0000081015958, 1471.0000012076148, 0.7592182246175333],
    [2213.000001524416, 1464.0000243454842, 0.7592182246175333],
    [2115.9999952456633, 1475.0000015176133, 0.7592182246175333],
    [2242.0000023736206, 1477.0000046142911, 0.7592182246175333],
    [2463.9999967564663, 1464.0000011704756, 0.7592182246175333],
    [2986.000011249705, 1540.9999994523619, 0.7592182246175333],
    [2760.00000317619, 1514.0000002442944, 0.7592182246175333],
    [3372.0000025298395, 1617.9999995345927, 0.7592182246175333],
    [3187.0000005152106, 1564.9999952212884, 0.7592182246175333],
    [3952.0000062252166, 1765.0000234029771, 0.7592182246175333],
    [200.99999875941003, 1190.0000046393075, 0.8545132017764238],
    [463.0000067425734, 1121.999995664854, 0.8545132017764238],
    [1455.0000001416358, 936.9999983034195, 0.8545132017764238],
    [1673.9999958962637, 927.9999993432831, 0.8545132017764238],
    [2492.0000021823594, 922.0000038312226, 0.8545132017764238],
    [2639.999994859976, 936.0000024781906, 0.8545132017764238],
    [3476.9999490636446, 1027.999983836245, 0.8545132017764238],
    [3638.9999965727247, 1088.0000258143732, 0.8545132017764238],
    [4002.0000051610787, 1149.9999925115812, 0.8545132017764238],
    [2296.9999822277705, 908.0000093918238, 0.8545132017764238],
    [266.00000015817864, 576.0000004915707, 0.9419541973013397],
    [364.00001493127616, 564.0000013624797, 0.9419541973013397],
    [752.9999995824019, 496.9999948653093, 0.9419541973013397],
    [845.9999975860665, 479.0000073040181, 0.9419541973013397],
    [1152.0000082161678, 421.9999937722655, 0.9419541973013397],
    [1215.0000019951258, 431.0001986750437, 0.9419541973013397],
    [1728.0000096657914, 368.0000024775422, 0.9419541973013397],
    [2095.9999932673395, 365.9999986230422, 0.9419541973013397],
    [2194.0000006543587, 356.99999967534075, 0.9419541973013397],
    [2598.0000021676074, 386.99999979901884, 0.9419541973013397],
    [2959.9998766657627, 410.0000032318384, 0.9419541973013397],
]

RING = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

DS = [4.15695, 2.93940753, 2.4000162, 2.078475, 1.85904456, 1.69706773, 1.46970377, 1.38565, 1.31454301, 1.25336758, 1.2000081, 1.15293049, 1.11099162, 1.0392375, 1.00820847, 0.97980251, 0.95366973, 0.92952228, 0.90712086, 0.88626472, 0.84853387, 0.83139, 0.81524497, 0.8000054, 0.77192624, 0.75895176, 0.73485188, 0.72363211, 0.71291104, 0.7026528, 0.692825, 0.68339837, 0.67434634, 0.65727151, 0.64920652, 0.64143131, 0.63392893, 0.62668379, 0.61968152, 0.61290884, 0.60000405, 0.59385, 0.58788151, 0.58208943, 0.57646525, 0.571001, 0.56568924, 0.55549581, 0.55060148, 0.54583428, 0.54118879, 0.53224291, 0.52793318, 0.52372647, 0.51961875, 0.51560619, 0.51168517, 0.50785227, 0.50410423, 0.50043797, 0.49685056]

def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name), arr)
    return [list(arr.shape), str(arr.dtype)]


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    data = np.array(DATA, dtype=np.float64)
    data[:, 2] = RING  # overwrite col2 with integer ring indices (as in the test)

    mycalibrant = cal.Calibrant(dspacing=list(DS), wavelength=WAVELENGTH)
    r = GeometryRefinement(
        data,
        pixel1=PIXEL1,
        pixel2=PIXEL2,
        wavelength=WAVELENGTH,
        calibrant=mycalibrant,
    )

    # --- Fixed parameter vector = the post-guess_poni initial estimate. ---
    fixed6 = np.array(
        [r.dist, r.poni1, r.poni2, r.rot1, r.rot2, r.rot3], dtype=np.float64
    )

    d1 = data[:, 0]
    d2 = data[:, 1]
    rings = data[:, 2]

    tth_meas = np.ascontiguousarray(r.tth(d1, d2, fixed6), dtype=np.float64)
    expected = np.ascontiguousarray(
        r.calc_2th(rings, r.wavelength), dtype=np.float64
    )
    residual = np.ascontiguousarray(
        r.residu1(fixed6, d1, d2, rings), dtype=np.float64
    )
    chi2_fixed = float(r.chi2(tuple(fixed6)))

    # Calibrant visible 2theta list at this wavelength (what calc_2th indexes).
    cal_2th = np.ascontiguousarray(mycalibrant.get_2th(), dtype=np.float64)
    cal_dspacing = np.ascontiguousarray(mycalibrant.dspacing, dtype=np.float64)

    meta = {}
    meta["control_points"] = save("control_points.npy", data)  # (51,3): d1,d2,ring
    meta["fixed_param6"] = save("fixed_param6.npy", fixed6)
    meta["tth_measured"] = save("tth_measured.npy", tth_meas)
    meta["expected_2th"] = save("expected_2th.npy", expected)
    meta["residual"] = save("residual.npy", residual)
    meta["calibrant_2th"] = save("calibrant_2th.npy", cal_2th)
    meta["calibrant_dspacing"] = save("calibrant_dspacing.npy", cal_dspacing)

    npt = data.shape[0]

    # --- Converged params / cost, TWO refine modes (both fmin_slsqp). ---
    # (a) refine2 = refine3(fix=["wavelength"]) -> rot3 FREE. For this dataset
    #     rot3 is a null direction of the cost surface (perturbing it barely
    #     changes chi2), so its converged value is optimizer-dependent; the Rust
    #     all-free refine is gated by COST against this, not per-parameter value.
    ra = GeometryRefinement(
        data, pixel1=PIXEL1, pixel2=PIXEL2, wavelength=WAVELENGTH,
        calibrant=cal.Calibrant(dspacing=list(DS), wavelength=WAVELENGTH),
    )
    ra.refine2(10000000)
    converged6 = np.array(
        [ra.dist, ra.poni1, ra.poni2, ra.rot1, ra.rot2, ra.rot3], dtype=np.float64
    )
    chi2_converged = float(ra.chi2())
    meta["converged_param6"] = save("converged_param6.npy", converged6)

    # (b) refine3(fix=["wavelength", "rot3"]) -> rot3 PINNED at its start. With
    #     the null direction removed the remaining 5-parameter minimum is unique,
    #     so the Rust rot3-fixed refine is gated tightly PER PARAMETER against
    #     this.
    rb = GeometryRefinement(
        data, pixel1=PIXEL1, pixel2=PIXEL2, wavelength=WAVELENGTH,
        calibrant=cal.Calibrant(dspacing=list(DS), wavelength=WAVELENGTH),
    )
    rb.refine3(10000000, fix=["wavelength", "rot3"])
    converged6_fix_rot3 = np.array(
        [rb.dist, rb.poni1, rb.poni2, rb.rot1, rb.rot2, rb.rot3], dtype=np.float64
    )
    chi2_converged_fix_rot3 = float(rb.chi2())
    meta["converged_param6_fix_rot3"] = save(
        "converged_param6_fix_rot3.npy", converged6_fix_rot3
    )

    manifest = {
        "dataset": "georef_noSpline_LaB6",
        "pyfai_version": pyFAI.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "config": {
            "pixel1": PIXEL1,
            "pixel2": PIXEL2,
            "wavelength": WAVELENGTH,
            "orientation": int(r.detector.orientation),
            "npt": int(npt),
            "n_rings": int(max(RING) + 1),
            "fixed_param6": list(fixed6),
            "chi2_fixed": chi2_fixed,
            "converged_param6": list(converged6),
            "chi2_converged": chi2_converged,
            "converged_param6_fix_rot3": list(converged6_fix_rot3),
            "chi2_converged_fix_rot3": chi2_converged_fix_rot3,
        },
        "arrays": meta,
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("wrote", OUTDIR)
    print("chi2_fixed              ", repr(chi2_fixed))
    print("chi2_converged (rot3 free)", repr(chi2_converged))
    print("converged6     (rot3 free)", list(converged6))
    print("chi2_converged (rot3 fix) ", repr(chi2_converged_fix_rot3))
    print("converged6     (rot3 fix) ", list(converged6_fix_rot3))


if __name__ == "__main__":
    main()
