#!/usr/bin/env python
"""Golden generator for the goniometer subsystem (rsfai-goniometer).

Run single-thread in the daq env (pyFAI 2026.5.0 built -ffp-contract=off,
scipy 1.17.1, numpy 2.4.3, numexpr present):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_goniometer.py

THREE parity surfaces, all feeding crates/rsfai-goniometer/tests/golden_goniometer.rs:

  * GeometryTransformation (the BIT-EXACT formula gate). A real 2-motor
    GeometryTransformation (one translation on dist, one rotation on rot2, plus a
    poni1 formula that exercises `**` and `sqrt`/`sin` so the numexpr-vs-Rust
    transcendental + integer-power path is verified, not assumed). We dump the
    six PONI outputs (numexpr-evaluated, float()) for several positions; the Rust
    expression evaluator must reproduce them bit-for-bit (f64 to_bits).

  * GoniometerRefinement residual / chi2 (the BIT-EXACT refinement gate). A
    GoniometerRefinement over two single geometries (LaB6 control points at two
    motor positions). At a FIXED goniometer parameter vector we dump residu2
    (the mean squared 2theta error). The Rust side recomputes residu2 from the
    same params/control-points/calibrant and must match (the geometry
    atan2/sin/cos is the only ULP-budgeted part).

  * Converged params / cost (the TOLERANCE gate). We run pyFAI's `refine2`
    (scipy SLSQP, Nelder-Mead-equivalent free vector) and dump the converged
    parameter vector and its cost. The Rust `refine()` (argmin Nelder-Mead) is
    compared at a recorded relative tolerance with cost_rust <= cost_pyfai.

  * MultiGeometryFiber 1D + 2D (the BIT-EXACT fiber-combine gate). Two
    FiberIntegrator frames combined via MultiGeometryFiber.integrate_fiber /
    integrate2d_fiber (direct per-bin accumulator summation, NOT the azimuthal
    union fold). We dump the combined intensity + summed accumulators; the Rust
    side reproduces them bit-for-bit (f64 accumulators, sequential left-fold).

Provenance (pyFAI/numpy/scipy/numexpr versions, thread count, formula strings,
fixed + converged params, fiber geometry) is in manifest.json.
"""

import json
import os
import platform
import warnings

import numpy as np
import scipy
import numexpr
import pyFAI
from pyFAI import calibrant as cal
from pyFAI.goniometer import GeometryTransformation, GoniometerRefinement
from pyFAI.detectors import detector_factory
from pyFAI.integrator.fiber import FiberIntegrator
from pyFAI.multi_geometry import MultiGeometryFiber

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_goniometer")

# --- Transformation A (FORMULA gate only): exercises the full numexpr op set
#   so the Rust expression evaluator is GOLDEN-verified, not assumed —
#   `**` (integer power), `sqrt`, `sin`, `pi`, division, nested parens. This is
#   used ONLY to evaluate the six PONI outputs bit-for-bit; it is not refined.
FORMULA_PARAM_NAMES = [
    "dist_scale", "dist_offset", "poni1_base", "poni2", "rot1",
    "rot2_scale", "rot2_offset", "bow",
]
FORMULA_POS_NAMES = ["pos_dist", "pos_angle"]
FORMULA_EXPRS = dict(
    dist_expr="pos_dist * dist_scale + dist_offset",
    poni1_expr="poni1_base + bow * sin(pos_angle) ** 2 + sqrt(bow) / pi",
    poni2_expr="poni2 + (rot1 - bow) ** 3",
    rot1_expr="rot1",
    rot2_expr="pos_angle * rot2_scale + rot2_offset",
    rot3_expr="0.0",
)
FORMULA_PARAM = [
    0.01,       # dist_scale
    0.20,       # dist_offset
    0.10,       # poni1_base
    0.11,       # poni2
    0.005,      # rot1
    0.0174533,  # rot2_scale (~1 deg per motor unit)
    0.001,      # rot2_offset
    0.0004,     # bow
]
POSITIONS = [
    [0.0, 0.0],
    [1.0, 5.0],
    [2.5, -3.0],
    [-1.5, 12.0],
    [0.3, 0.7],
]

# --- Transformation B (REFINEMENT gate): a well-posed goniometer whose six
#   params ARE the six pyFAI geometry params (identity passthrough per
#   component). A single motor `pos` is bound but referenced only via a
#   `0.0 * pos` term in rot2 so the motor-binding plumbing is exercised without
#   adding a null fit direction. This mirrors rsfai-calib's well-posed
#   test_noSpline refinement (chi2 ~1.25e-2 -> ~7.35e-4), so SLSQP converges
#   cleanly (status 0) on the five ring-constrained params; rot3 is a null
#   direction and is left near its start.
PARAM_NAMES = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
POS_NAMES = ["pos"]
DIST_EXPR = "dist"
PONI1_EXPR = "poni1"
PONI2_EXPR = "poni2"
ROT1_EXPR = "rot1"
ROT2_EXPR = "rot2 + 0.0 * pos"
ROT3_EXPR = "rot3"

# The "fixed" goniometer parameter vector (bit-exact residual gate + refine
# start): the test_noSpline post-`guess_poni` initial estimate (the same start
# rsfai-calib's golden uses).
FIXED_PARAM = [
    0.07024175,   # dist
    0.02680055,   # poni1
    0.05515133,   # poni2
    -0.29019663,  # rot1
    0.03717258,   # rot2
    0.0,          # rot3 (a null direction of ring data; stays ~0)
]

PIXEL = 1.5e-5
WAVELENGTH = 1.54e-10

# --- pyFAI test_geometry_refinement.py::test_noSpline LaB6 control points --
# (verbatim, the same fixture rsfai-calib uses; first 25 rows for a compact
#  two-geometry refinement.)
CONTROL_POINTS = [
    [1585.9999996029055, 2893.999999119241, 0.5300564938306779],
    [1853.9999932086102, 2873.000000163791, 0.5300564938306779],
    [2163.9999987531855, 2854.9999987738884, 0.5300564938306779],
    [2699.999997791493, 2893.9999985831755, 0.5300564938306779],
    [3186.9999966428777, 3028.9999985930604, 0.5300564938306779],
    [1561.0000027706968, 2627.000000529364, 0.7521953923842869],
    [1820.9999979673413, 2588.9999996158557, 0.7521953923842869],
    [2143.999999081593, 2562.9999990503972, 0.7521953923842869],
    [2706.9999983093525, 2585.9999992923594, 0.7521953923842869],
    [3210.99999698136, 2719.0000003219736, 0.7521953923842869],
    [1539.0000001229334, 2375.000000406348, 0.9298780331272518],
    [1796.0000023167762, 2326.000000284083, 0.9298780331272518],
    [2125.0000010972064, 2293.000000503300, 0.9298780331272518],
    [2700.0000010876255, 2306.0000010888773, 0.9298780331272518],
    [3232.0000016877327, 2440.0000002542136, 0.9298780331272518],
    [1517.9999992407241, 2123.000000123723, 1.0825147517738019],
    [1772.0000003839145, 2065.999999990455, 1.0825147517738019],
    [2106.9999980354752, 2024.9999990049748, 1.0825147517738019],
    [2693.000001076276, 2026.0000017288656, 1.0825147517738019],
    [3252.000001976179, 2160.999999990745, 1.0825147517738019],
    [1499.0000010839487, 1870.9999991683347, 1.2187859061771321],
    [1750.0000026536995, 1805.0000019021847, 1.2187859061771321],
    [2090.0000003820063, 1757.0000008539542, 1.2187859061771321],
    [2685.000000523629, 1746.0000010668696, 1.2187859061771321],
    [3270.0000018384985, 1879.9999996170894, 1.2187859061771321],
]

# Integer ring indices for the 25 control points (five rings of five points,
# 0-based). The third column of the verbatim test_noSpline fixture above is the
# *2theta in radians*, not a ring index; pyFAI's GeometryRefinement truncates
# `data[:, 2].astype(int32)` so that raw column silently collapses several rings
# to 0/1. gen_golden_calib.py does the same overwrite (`data[:, 2] = RING`) so
# the calibrant `calc_2th` list is indexed by the real ring index. Mirror it here
# so the residu2 gate matches the Rust side, which indexes `calc_2th` by the
# integer ring of each control point.
RING = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name), arr)
    return list(arr.shape), str(arr.dtype)


def pos_function(metadata):
    """The goniometer position is the metadata itself (a [pos_dist, pos_angle])."""
    return list(metadata)


def build_formula_transformation():
    return GeometryTransformation(
        param_names=FORMULA_PARAM_NAMES,
        pos_names=FORMULA_POS_NAMES,
        **FORMULA_EXPRS,
    )


def build_transformation():
    return GeometryTransformation(
        dist_expr=DIST_EXPR,
        poni1_expr=PONI1_EXPR,
        poni2_expr=PONI2_EXPR,
        rot1_expr=ROT1_EXPR,
        rot2_expr=ROT2_EXPR,
        rot3_expr=ROT3_EXPR,
        param_names=PARAM_NAMES,
        pos_names=POS_NAMES,
    )


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- Surface 1: GeometryTransformation outputs (bit-exact) -----------
    # Transformation A exercises the full numexpr op set (`**`, sqrt, sin, pi,
    # division, nested parens). poni_outputs[i] = [dist,poni1,poni2,rot1,rot2,rot3].
    gtf = build_formula_transformation()
    poni_outputs = []
    for pos in POSITIONS:
        ponip = gtf(FORMULA_PARAM, pos)
        poni_outputs.append([float(x) for x in ponip])
    save("transform_outputs.npy", np.array(poni_outputs, dtype=np.float64))
    save("transform_positions.npy", np.array(POSITIONS, dtype=np.float64))
    save("formula_param.npy", np.array(FORMULA_PARAM, dtype=np.float64))

    # ---- Surfaces 2 & 3: GoniometerRefinement residu2 + refine ----------
    save("fixed_param.npy", np.array(FIXED_PARAM, dtype=np.float64))
    gt = build_transformation()
    calibrant = cal.get_calibrant("LaB6")
    calibrant.wavelength = WAVELENGTH
    # Dump the calibrant d-spacing list (the raw `.dspacing` attribute, Angstrom)
    # so the Rust verifier rebuilds the exact same Calibrant — the residu2
    # bit-exact gate needs the calc_2th list to match pyFAI's LaB6 resource
    # bit-for-bit, not a hand-derived a/sqrt(n). Same convention as gen_golden_calib.
    dspacing = np.ascontiguousarray(calibrant.dspacing, dtype=np.float64)
    save("calibrant_dspacing.npy", dspacing)
    detector = detector_factory("Detector", {"pixel1": PIXEL, "pixel2": PIXEL,
                                             "max_shape": (4000, 4000)})

    gonioref = GoniometerRefinement(
        param=list(FIXED_PARAM),
        pos_function=pos_function,
        trans_function=gt,
        detector=detector,
        wavelength=WAVELENGTH,
        param_names=PARAM_NAMES,
        pos_names=POS_NAMES,
    )

    cp = np.array(CONTROL_POINTS, dtype=np.float64)
    cp[:, 2] = RING  # overwrite the 2theta column with integer ring indices.
    # Two single geometries sharing the same physical geometry (the test_noSpline
    # rings, split in two), both at motor pos=0 (the `0.0 * pos` term makes the
    # PONI pos-independent). Splitting across geometries (vs one) exercises the
    # cross-geometry residu2 accumulation while staying the well-posed single-PONI
    # fit the control points describe.
    geom_meta = [[0.0], [0.0]]
    halves = [cp[:13], cp[13:]]
    for label, (meta, pts) in enumerate(zip(geom_meta, halves)):
        ai = gonioref.get_ai(pos_function(meta))
        sg = gonioref.new_geometry(
            label=f"g{label}",
            metadata=meta,
            control_points=None,
            calibrant=calibrant,
            geometry=ai,
        )
        # Inject the control points directly into the geometry refinement.
        sg.geometry_refinement.data = np.ascontiguousarray(pts, dtype=np.float64)
        sg.control_points = None

    residu2_fixed = float(gonioref.residu2(np.asarray(FIXED_PARAM, dtype=np.float64)))

    # Refine (scipy SLSQP). refine2 mutates gonioref.param in place on success.
    converged = gonioref.refine2(method="slsqp")
    converged = [float(x) for x in converged]
    cost_converged = float(gonioref.residu2(np.asarray(converged, dtype=np.float64)))
    save("converged_param.npy", np.array(converged, dtype=np.float64))

    # ---- Surface 4: MultiGeometryFiber 1D + 2D (bit-exact) --------------
    # pyFAI's MultiGeometryFiber shares ONE set of grazing-incidence params
    # (incident_angle / tilt_angle / sample_orientation) across all frames — the
    # GI geometry is baked into the common ip/oop units. The frames differ by
    # their detector PONI geometry (as a goniometer arm would move the detector)
    # and by their data. So: two frames with distinct PONI geometries, one shared
    # GI param set, common units + bins.
    fiber_geoms = [
        dict(dist=0.20, poni1=0.10, poni2=0.10, rot1=0.01, rot2=0.02, rot3=0.0, wavelength=1e-10),
        dict(dist=0.22, poni1=0.11, poni2=0.09, rot1=0.00, rot2=0.05, rot3=0.0, wavelength=1e-10),
    ]
    SHARED_GI = dict(sample_orientation=1, incident_angle=0.2, tilt_angle=0.0)
    npt_ip, npt_oop = 200, 200
    unit_ip, unit_oop = "qip_nm^-1", "qoop_nm^-1"

    fdet = detector_factory("Pilatus1M")
    rng = np.random.default_rng(20260529)
    lst_data = [(rng.random(fdet.shape) * 1000.0).astype(np.float32) for _ in fiber_geoms]
    for j, d in enumerate(lst_data):
        save(f"fiber_data_{j}.npy", d)

    fis = []
    for j, fg in enumerate(fiber_geoms):
        fi = FiberIntegrator(detector=detector_factory("Pilatus1M"), **fg)
        # Save each frame's .poni so the Rust side rebuilds the identical geometry.
        fi.save(os.path.join(OUTDIR, f"fiber_geometry_{j}.poni"))
        fis.append(fi)

    # One shared GI param set; let the MGF guess the common ip/oop range
    # (it concatenates the two frames' position ranges, so both bin on one axis).
    # The ip/oop UNITS come from the MGF constructor; passing them again to
    # integrate*_fiber collides through pyFAI's **kwargs forwarding (a 2D bug),
    # so only npt/method are passed here. Serialize (threadpoolsize=0) for the
    # single-thread deterministic golden — the ThreadPool path also segfaults the
    # rebuilt cython here.
    mgf = MultiGeometryFiber(
        fis, unit=(unit_ip, unit_oop),
        incident_angle=SHARED_GI["incident_angle"],
        tilt_angle=SHARED_GI["tilt_angle"],
        sample_orientation=SHARED_GI["sample_orientation"],
        empty=0.0,
        threadpoolsize=0,
    )
    res1d = {}
    res2d = {}
    for vert in (True, False):
        vtag = "v" if vert else "h"
        r = mgf.integrate_fiber(
            lst_data, npt_ip=npt_ip, npt_oop=npt_oop,
            vertical_integration=vert, correctSolidAngle=True,
            method=("no", "histogram", "cython"),
        )
        with warnings.catch_warnings():
            # `radial` warns for a fiber result (deprecated alias of the
            # integrated axis); the value is the bin centres we compare.
            warnings.simplefilter("ignore")
            axis = np.asarray(r.radial)
        res1d[vtag] = {
            "axis": save(f"mgf_1d{vtag}__axis.npy", axis),
            "intensity": save(f"mgf_1d{vtag}__intensity.npy", np.asarray(r.intensity)),
            "sum_signal": save(f"mgf_1d{vtag}__sum_signal.npy", np.asarray(r.sum_signal)),
            "sum_normalization": save(f"mgf_1d{vtag}__sum_normalization.npy",
                                      np.asarray(r.sum_normalization)),
            "count": save(f"mgf_1d{vtag}__count.npy", np.asarray(r.count)),
        }

    r2 = mgf.integrate2d_fiber(
        lst_data, npt_ip=npt_ip, npt_oop=npt_oop,
        correctSolidAngle=True, method=("no", "histogram", "cython"),
    )
    res2d = {
        "inplane": save("mgf_2d__inplane.npy", np.asarray(r2.radial)),
        "outofplane": save("mgf_2d__outofplane.npy", np.asarray(r2.azimuthal)),
        "intensity": save("mgf_2d__intensity.npy", np.asarray(r2.intensity)),
        "sum_signal": save("mgf_2d__sum_signal.npy", np.asarray(r2.sum_signal)),
        "sum_normalization": save("mgf_2d__sum_normalization.npy",
                                  np.asarray(r2.sum_normalization)),
        "count": save("mgf_2d__count.npy", np.asarray(r2.count)),
    }

    manifest = {
        "dataset": "goniometer",
        "source": "gen_golden_goniometer.py",
        "versions": {
            "pyFAI": pyFAI.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numexpr": numexpr.__version__,
            "python": platform.python_version(),
        },
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "transformation": {
            "param_names": PARAM_NAMES,
            "pos_names": POS_NAMES,
            "dist_expr": DIST_EXPR,
            "poni1_expr": PONI1_EXPR,
            "poni2_expr": PONI2_EXPR,
            "rot1_expr": ROT1_EXPR,
            "rot2_expr": ROT2_EXPR,
            "rot3_expr": ROT3_EXPR,
        },
        "config": {
            "pixel1": PIXEL,
            "pixel2": PIXEL,
            "wavelength": WAVELENGTH,
            "orientation": int(detector.orientation.value),
            "geom_meta": geom_meta,
            "residu2_fixed": residu2_fixed,
            "cost_converged": cost_converged,
            "n_control_points": [int(h.shape[0]) for h in halves],
        },
        "fiber": {
            "detector": "Pilatus1M",
            "geometries": fiber_geoms,
            "shared_gi": SHARED_GI,
            "npt_ip": npt_ip,
            "npt_oop": npt_oop,
            "unit_ip": unit_ip,
            "unit_oop": unit_oop,
            "result1d": res1d,
            "result2d": res2d,
        },
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote goniometer golden to {OUTDIR}")
    print(f"  residu2 at fixed param = {residu2_fixed:.12e}")
    print(f"  converged cost         = {cost_converged:.12e}")
    print(f"  converged param        = {converged}")


if __name__ == "__main__":
    main()
