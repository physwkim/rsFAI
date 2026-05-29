#!/usr/bin/env python3
"""Validate the rsFAI backend's *OpenCL* routing against the committed OpenCL
golden datasets, in-process.

Run in the ``daq`` conda env (pyFAI + pyopencl + the maturin-built ``rsfai``):

    /Users/stevek/mamba/envs/daq/bin/python golden/test_backend_gpu.py

For every dataset with an ``opencl_params.json`` (method impl ``opencl``) this
loads the PONI, ``rsfai_backend.install``s the backend, and runs
``integrate1d_ng`` / ``integrate2d`` with the manifest's ``("split", algo,
"opencl")`` method, then checks each field against the committed golden (which
pyFAI's OpenCL integrators produced on this same device).

Routing assertions — the structural point of the test:
  * ``algo`` ∈ {csr, lut}: the backend MUST take rsFAI's ``GpuEngine`` path
    (``compute_engine`` starts with ``rsfai``).  Gate: relative error <= 1e-6
    (CSR/LUT are deterministic; bit-exact in practice on this device).
  * ``algo`` == histogram: the backend MUST fall back to pyFAI's OpenCL
    (``compute_engine`` does NOT start with ``rsfai``).  Gate: relative error
    <= 5e-5 (atomic-scatter noise); ``count`` is integer atomics (exact).

Exit code 0 only if every routing assertion holds and every field is in gate.
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
CSR_LUT_TOL = 1e-6   # deterministic sparse reduce
HIST_TOL = 5e-5      # atomic-scatter histogram noise

AXIS_FIELDS = ("radial", "azimuthal")
ACCUMULATOR_FIELDS = (
    "intensity", "sigma", "count", "sum_signal", "sum_variance",
    "sum_normalization", "sum_normalization2", "std", "sem",
)


def load(d, name):
    return np.load(DATASETS / d / f"{name}.npy")


def relerr(got, ref):
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if got.shape != ref.shape:
        return float("inf")
    denom = np.maximum(np.abs(ref), 1e-30)
    return float(np.max(np.abs(got - ref) / denom))


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


def dataset_dirs():
    return sorted(
        p.name for p in DATASETS.iterdir()
        if (p / "opencl_params.json").exists() and (p / "manifest.json").exists()
        and (p / "geometry.poni").exists()
    )


def main():
    print(f"pyFAI {pyFAI.version} | rsfai_backend OpenCL routing | "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"numpy {np.__version__} | csr/lut gate {CSR_LUT_TOL:.0e}, "
          f"histogram-fallback gate {HIST_TOL:.0e}\n")

    total_checked = 0
    total_fail = 0
    n_gpu = 0
    n_fallback = 0

    for d in dataset_dirs():
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        algo = cfg["method"][1]
        gpu_algo = algo in ("csr", "lut")
        tol = CSR_LUT_TOL if gpu_algo else HIST_TOL

        print(f"=== {d} ===")
        fields, engine = run_backend(d, cfg)
        used_rsfai = engine.startswith("rsfai")
        print(f"  engine: {engine}  [{'rsFAI GPU' if used_rsfai else 'pyFAI fallback'}]")

        # Routing assertion: csr/lut -> rsFAI GPU; histogram -> pyFAI fallback.
        total_checked += 1
        route_ok = used_rsfai if gpu_algo else (not used_rsfai)
        if not route_ok:
            total_fail += 1
            want = "rsFAI GPU" if gpu_algo else "pyFAI fallback"
            print(f"  routing            FAIL  | algo={algo} expected {want}")
        else:
            n_gpu += int(gpu_algo)
            n_fallback += int(not gpu_algo)
            print(f"  routing            PASS  | algo={algo}")

        for field in AXIS_FIELDS + ACCUMULATOR_FIELDS:
            gpath = DATASETS / d / f"out_{field}.npy"
            if not gpath.exists():
                continue
            golden = np.load(gpath)
            total_checked += 1
            if field not in fields:
                total_fail += 1
                print(f"  out_{field:20s} FAIL  | backend did not expose field")
                continue
            # count is integer atomics in the histogram path -> exact there.
            field_tol = 0.0 if (field == "count" and not gpu_algo) else tol
            e = relerr(fields[field], golden)
            ok = e <= field_tol
            total_fail += (not ok)
            gate = "exact" if field_tol == 0.0 else f"rel<={field_tol:.0e}"
            print(f"  out_{field:20s} {'PASS' if ok else 'FAIL'} [{gate:>10s}] | rel_err={e:.3e}")
        print()

    print(f"datasets: {n_gpu} via rsFAI GPU, {n_fallback} via pyFAI fallback")
    print(f"checked {total_checked} assertions, {total_fail} failed")
    if total_fail:
        print("RESULT: FAIL — see per-field detail above")
        return 1
    print("RESULT: PASS — backend OpenCL routing reproduces golden within gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
