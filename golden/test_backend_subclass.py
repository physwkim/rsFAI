#!/usr/bin/env python3
"""Validate ``rsfai_backend.RsfaiAzimuthalIntegrator`` (the pyFAI subclass whose
integration runs on rsFAI) against the committed golden datasets, in-process.

Run in the ``daq`` conda env (pyFAI + the maturin-built ``rsfai``):

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/test_backend_subclass.py

For each non-OpenCL dataset this:
  * loads the PONI with ``pyFAI.load`` and ``rsfai_backend.install``s the rsFAI
    backend onto that genuine pyFAI integrator;
  * runs ``integrate1d_ng`` / ``integrate2d`` exactly as the GUI would (method,
    unit, error_model, ranges, polarization, normalization from the manifest);
  * checks every exposed result field against the committed golden, AND against
    a plain (un-patched) pyFAI run, so by transitivity backend == live pyFAI.

The gate mirrors ``golden/test_inprocess_dropin.py``: bit-exact for the serial
engines, relative error <= 1e-6 for the rayon-parallel no-split 1D histogram.
``result.compute_engine`` tells whether the rsFAI path ran or it fell back to
pyFAI (e.g. range/opencl configs); a fallback must still match golden bit-exact.
Exit code 0 only if every field of every dataset passes its gate.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyFAI  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsfai_backend  # noqa: E402

DATASETS = HERE / "datasets"
REL_TOL = 1e-6

ACCUMULATOR_FIELDS = (
    "intensity", "sigma", "count", "sum_signal", "sum_variance",
    "sum_normalization", "sum_normalization2", "std", "sem",
)
AXIS_FIELDS = ("radial", "azimuthal")


def load(d, name):
    return np.load(DATASETS / d / f"{name}.npy")


def _max_rel(a, g):
    a = np.ascontiguousarray(a).ravel().astype(np.float64)
    g = np.ascontiguousarray(g).ravel().astype(np.float64)
    if a.shape != g.shape:
        return float("inf"), 0
    both_nan = np.isnan(a) & np.isnan(g)
    one_nan = np.isnan(a) ^ np.isnan(g)
    n_nan = int(np.count_nonzero(one_nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.abs(a - g)
        rel = np.where(g != 0.0, diff / np.abs(g), np.where(a == g, 0.0, np.inf))
    rel = np.where(both_nan, 0.0, rel)
    rel = np.where(one_nan, np.inf, rel)
    return float(np.max(rel)) if rel.size else 0.0, n_nan


def compare(actual, golden):
    """(bit_exact, max_rel, detail). dtype/shape mismatch => hard fail."""
    a = np.ascontiguousarray(actual)
    g = np.ascontiguousarray(golden)
    if a.dtype != g.dtype:
        return False, float("inf"), f"dtype {a.dtype} != {g.dtype}"
    if a.shape != g.shape:
        return False, float("inf"), f"shape {a.shape} != {g.shape}"
    bit = a.tobytes() == g.tobytes()
    if bit:
        return True, 0.0, "bit-exact"
    max_rel, n_nan = _max_rel(a, g)
    return False, max_rel, f"max_rel={max_rel:.2e}" + (f", {n_nan} NaN-bit" if n_nan else "")


def run_backend(d, cfg):
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    rsfai_backend.install(ai)
    img = load(d, "image")
    common = dict(
        unit=cfg["unit"], method=tuple(cfg["method"]),
        correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"],
        radial_range=cfg["radial_range"], azimuth_range=cfg["azimuth_range"],
    )
    if cfg.get("dim", 1) == 2:
        res = ai.integrate2d(img, cfg["npt_rad"], cfg["npt_azim"], **common)
    else:
        res = ai.integrate1d_ng(img, cfg["npt"], **common)
    fields = {}
    for a in AXIS_FIELDS + ACCUMULATOR_FIELDS:
        v = getattr(res, a, None)
        if isinstance(v, np.ndarray):
            fields[a] = v
    return fields, str(res.compute_engine)


def run_live(d, cfg):
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = load(d, "image")
    common = dict(
        unit=cfg["unit"], method=tuple(cfg["method"]),
        correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"],
        radial_range=cfg["radial_range"], azimuth_range=cfg["azimuth_range"],
    )
    if cfg.get("dim", 1) == 2:
        res = ai.integrate2d_ng(img, cfg["npt_rad"], cfg["npt_azim"], **common)
    else:
        res = ai.integrate1d_ng(img, cfg["npt"], **common)
    out = {}
    for a in AXIS_FIELDS + ACCUMULATOR_FIELDS:
        v = getattr(res, a, None)
        if isinstance(v, np.ndarray):
            out[a] = v
    return out


def dataset_dirs():
    return sorted(
        p.name for p in DATASETS.iterdir()
        if (p / "manifest.json").exists() and not (p / "opencl_params.json").exists()
    )


def main():
    print(f"pyFAI {pyFAI.version} | rsfai_backend.RsfaiAzimuthalIntegrator | "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"numpy {np.__version__} | rel gate {REL_TOL:.0e} for the no-split 1D histogram\n")

    total_checked = 0
    total_fail = 0
    n_rsfai = 0
    n_fallback = 0

    for d in dataset_dirs():
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        dim = cfg.get("dim", 1)
        split, algo = cfg["method"][0], cfg["method"][1]
        # The no-split 1D histogram is rayon-parallel (non-deterministic f64 add
        # order); every other engine is serial -> bit-exact.
        acc_exact = not (dim == 1 and (split, algo) == ("no", "histogram"))

        print(f"=== {d} ===")
        b_fields, engine = run_backend(d, cfg)
        used_rsfai = engine.startswith("rsfai")
        if used_rsfai:
            n_rsfai += 1
        else:
            n_fallback += 1
            # A fallback ran pyFAI: it must reproduce golden exactly on every field.
            acc_exact = True
        live_fields = run_live(d, cfg)
        print(f"  engine: {engine}  [{'rsFAI' if used_rsfai else 'pyFAI fallback'}]")

        for field in AXIS_FIELDS + ACCUMULATOR_FIELDS:
            gpath = DATASETS / d / f"out_{field}.npy"
            if not gpath.exists():
                continue
            golden = np.load(gpath)
            is_axis = field in AXIS_FIELDS
            exact = is_axis or acc_exact

            if field not in b_fields:
                # pyFAI exposes this field (golden has it) but the backend did not.
                total_checked += 1
                total_fail += 1
                print(f"  out_{field:20s} FAIL  | backend did not expose field")
                continue

            b_bit, b_rel, b_detail = compare(b_fields[field], golden)
            b_ok = b_bit if exact else (b_rel <= REL_TOL)

            l = live_fields.get(field)
            if l is None:
                l_ok, l_detail = False, "live pyFAI did not expose field"
            else:
                l_bit, _, l_detail = compare(l, golden)
                l_ok = l_bit

            total_checked += 2
            total_fail += (not b_ok) + (not l_ok)
            gate = "exact" if exact else f"rel<={REL_TOL:.0e}"
            status = "PASS" if (b_ok and l_ok) else "FAIL"
            print(f"  out_{field:20s} {status} [{gate:>10s}] | backend: {b_detail} | live: {l_detail}")
        print()

    print(f"datasets: {n_rsfai} via rsFAI, {n_fallback} via pyFAI fallback")
    print(f"checked {total_checked} comparisons, {total_fail} failed")
    if total_fail:
        print("RESULT: FAIL — see per-field detail above")
        return 1
    print("RESULT: PASS — RsfaiAzimuthalIntegrator == golden == live pyFAI on every field")
    return 0


if __name__ == "__main__":
    sys.exit(main())
