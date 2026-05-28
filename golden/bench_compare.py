#!/usr/bin/env python3
"""Steady-state per-frame performance: pyFAI vs rsFAI, every implemented CPU tuple.

Run in the ``daq`` conda env (both ``pyFAI`` and the maturin-built ``rsfai``).
Threading is controlled ENTIRELY by the environment so the same script gives the
single-thread and multi-thread tables; this script never sets OMP/RAYON itself:

    # single-thread (pure algorithm comparison)
    OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        /Users/stevek/mamba/envs/daq/bin/python golden/bench_compare.py

    # multi-thread (real-world throughput; both libraries use all cores)
    /Users/stevek/mamba/envs/daq/bin/python golden/bench_compare.py

What is measured (the user-chosen metric): STEADY-STATE per-frame cost, i.e. the
geometry arrays and the sparse matrix are built ONCE (untimed setup), and only the
per-frame work (preprocessing + binning/apply) is timed. This is the streaming-DAQ
number and it is the only fair one, because:

  * pyFAI's ``integrate1d_ng``/``integrate2d_ng`` cache the built integrator on the
    ``ai`` object (``ai.engines``), so after one warm-up call each subsequent call
    reuses the matrix and the normalized corrections -> per-frame = preproc + apply.
  * rsFAI's high-level ``AzimuthalIntegrator.integrate1d`` REBUILDS geometry + matrix
    every call (no cache), so timing it would unfairly charge build cost per frame.
    We therefore drive rsFAI's low-level kernels: build the matrix once (using the
    same dumped geometry the parity harness trusts, bit-identical to pyFAI's), then
    time ``rsfai.preproc4(image) + <apply>`` per frame.

Caveats, stated plainly:
  * pyFAI numbers are the high-level ``integrate*_ng`` (includes pyFAI's per-call
    Python orchestration: arg parsing, engine lookup, normalization checks). pyFAI's
    low-level cython apply is not callable standalone, so this is the realistic and
    only-available pyFAI per-frame figure.
  * rsFAI numbers are the low-level kernel path (``preproc4`` + apply), minimal Python
    overhead. Both sides do preproc + apply on the SAME float32 image with the SAME
    precomputed solid-angle/polarization arrays, so the compute work is matched.
  * ``pseudo`` (2D) has no rsFAI port, so it is reported pyFAI-only.

One canonical config per tuple: Pilatus1M, q_nm^-1 (CSR 1D golden only exists in
2th_deg -> used there; the radial unit does not affect per-frame cost), npt=1000
(1D) / npt_rad=100 x npt_azim=36 (2D), error_model=poisson, solid-angle on,
polarization_factor=0.99, no range.
"""

import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pyFAI
import rsfai

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
TWO_PI = 2.0 * math.pi

TUPLES = [
    ("no", "histogram"), ("bbox", "histogram"), ("full", "histogram"),
    ("no", "csr"), ("bbox", "csr"), ("full", "csr"),
    ("no", "csc"), ("bbox", "csc"), ("full", "csc"),
    ("no", "lut"), ("bbox", "lut"), ("full", "lut"),
]
# pyFAI-only (no rsFAI port):
PYFAI_ONLY = [("pseudo", "histogram")]  # 2D only


def load(d, name):
    return np.load(DATASETS / d / f"{name}.npy")


def find_dataset(split, algo, dim):
    npttag = "npt100x36" if dim == 2 else "npt1000"
    cands = []
    for p in DATASETS.iterdir():
        n = p.name
        if not n.startswith(f"Pilatus1M__{split}-{algo}-cython__"):
            continue
        if npttag not in n or "errpoisson" not in n:
            continue
        if any(x in n for x in ("razim", "rrad", "orient", "opencl")):
            continue
        if not (p / "manifest.json").exists():
            continue
        cfg = json.load(open(p / "manifest.json"))["config"]
        if cfg.get("dim", 1) != dim:
            continue
        cands.append(n)
    cands.sort(key=lambda n: (0 if "q_nmm1" in n else 1, n))
    return cands[0] if cands else None


def time_call(fn, reps, warmup):
    """Return per-call wall times in seconds (median/min computed by caller)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


# ----------------------------------------------------------------------------
# pyFAI: warm-cached integrate*_ng (per-frame = preproc + apply, matrix cached)
# ----------------------------------------------------------------------------
def make_pyfai_fn(d, cfg):
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = np.ascontiguousarray(load(d, "image").astype(np.float32))
    common = dict(
        unit=cfg["unit"], method=tuple(cfg["method"]),
        correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"],
    )
    if cfg.get("dim", 1) == 2:
        npt_rad, npt_azim = cfg["npt_rad"], cfg["npt_azim"]
        return lambda: ai.integrate2d_ng(img, npt_rad, npt_azim, **common)
    npt = cfg["npt"]
    return lambda: ai.integrate1d_ng(img, npt, **common)


# ----------------------------------------------------------------------------
# rsFAI: build matrix ONCE, then time preproc4 + apply per frame
# ----------------------------------------------------------------------------
def make_rsfai_fn(d, cfg):
    """Return (preproc_fn, apply_fn, build_seconds). Matrix/geometry built untimed.

    preproc_fn() -> prep (the per-frame preprocessing; identical across every tuple
    of a given dim). apply_fn(prep) -> result (the binning, reusing the prebuilt
    matrix). The realistic low-level per-frame cost is apply_fn(preproc_fn()); the
    two are also timed separately so the prep round-trip (a 16MB numpy array that
    pyFAI's fused C path keeps internal) can be told apart from the binning kernel."""
    dim = cfg.get("dim", 1)
    split, algo = cfg["method"][0], cfg["method"][1]
    em = cfg["error_model_code"]
    cdp = cfg.get("chi_disc_at_pi", True)   # 2D-only manifest keys; 1D builders
    p1p = cfg.get("pos1_period", TWO_PI)     # take their own hardcoded defaults

    image = np.ascontiguousarray(load(d, "image").astype(np.float32).reshape(-1))
    mask = np.ascontiguousarray(load(d, "mask").reshape(-1)).astype(np.int8)
    sa = (np.ascontiguousarray(load(d, "solidangle").astype(np.float32).reshape(-1))
          if cfg["correct_solid_angle"] else None)
    pol = (np.ascontiguousarray(load(d, "polarization").astype(np.float32).reshape(-1))
           if cfg["polarization_factor"] is not None else None)
    dummy = cfg["dummy"]
    preproc_kw = dict(
        solidangle=sa, polarization=pol, mask=mask,
        normalization_factor=np.float32(cfg["normalization_factor"]),
        poissonian=(em == 2), check_dummy=(dummy is not None),
        dummy=float(dummy if dummy is not None else 0.0),
        delta_dummy=float(cfg["delta_dummy"] if cfg["delta_dummy"] is not None else 0.0),
    )

    pos0 = np.ascontiguousarray(load(d, "pos0_center_unscaled").reshape(-1))
    dpos0 = np.ascontiguousarray(load(d, "pos0_delta").reshape(-1))
    chi = np.ascontiguousarray(load(d, "chi_center").reshape(-1))
    dchi = np.ascontiguousarray(load(d, "chi_delta").reshape(-1))
    corners = np.ascontiguousarray(load(d, "corners").astype(np.float64).reshape(-1))
    npt = cfg["npt"] if dim == 1 else (cfg["npt_rad"], cfg["npt_azim"])

    t_build = time.perf_counter()

    # ---- histogram (no matrix to reuse; binning runs each frame) ----
    if algo == "histogram":
        if dim == 1:
            if split == "no":
                apply = lambda prep: rsfai.histogram1d(pos0, prep, npt, error_model=em, empty=0.0)
            elif split == "bbox":
                apply = lambda prep: rsfai.histogram1d_bbox(
                    pos0, dpos0, prep, mask=mask, npt=npt, error_model=em,
                    empty=0.0, allow_pos0_neg=False)
            else:  # full
                apply = lambda prep: rsfai.histogram1d_full(
                    corners, prep, mask=mask, npt=npt, error_model=em, empty=0.0,
                    allow_pos0_neg=False, chi_disc_at_pi=True, pos1_period=TWO_PI)
        else:
            if split == "no":
                apply = lambda prep: rsfai.histogram2d(
                    pos0, chi, prep, bins=npt, mask=mask, error_model=em,
                    allow_radial_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p, empty=0.0)
            elif split == "bbox":
                apply = lambda prep: rsfai.histogram2d_bbox(
                    pos0, dpos0, chi, dchi, prep, bins=npt, mask=mask, allow_pos0_neg=False,
                    chi_disc_at_pi=cdp, pos1_period=p1p, error_model=em, empty=0.0)
            else:  # full
                apply = lambda prep: rsfai.histogram2d_full(
                    corners, prep, bins=npt, mask=mask, allow_pos0_neg=False,
                    chi_disc_at_pi=cdp, pos1_period=p1p, error_model=em, empty=0.0)
        build_s = time.perf_counter() - t_build
        return (lambda: rsfai.preproc4(image, **preproc_kw)), apply, build_s

    # ---- sparse: build matrix ONCE, apply reuses it ----
    do_split = split == "bbox"
    if algo == "lut":
        if split in ("no", "bbox"):
            if dim == 1:
                idx, coef, lsz, bc = rsfai.build_bbox_lut_1d(
                    pos0, delta_pos0=(dpos0 if do_split else None), mask=mask,
                    bins=npt, allow_pos0_neg=False)
            else:
                idx, coef, lsz, bc0, bc1 = rsfai.build_bbox_lut_2d(
                    pos0, chi, delta_pos0=(dpos0 if do_split else None),
                    delta_pos1=(dchi if do_split else None), mask=mask, bins=npt,
                    allow_pos0_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p)
        else:  # full
            if dim == 1:
                idx, coef, lsz, bc = rsfai.build_full_lut_1d(
                    corners, mask=mask, bins=npt, allow_pos0_neg=False,
                    chi_disc_at_pi=True, pos1_period=TWO_PI)
            else:
                idx, coef, lsz, bc0, bc1 = rsfai.build_full_lut_2d(
                    corners, mask=mask, bins=npt, allow_pos0_neg=False,
                    chi_disc_at_pi=cdp, pos1_period=p1p)
        if dim == 1:
            apply = lambda prep: rsfai.lut_integrate1d(idx, coef, lsz, prep, bc, error_model=em, empty=0.0)
        else:
            apply = lambda prep: rsfai.lut_integrate2d(idx, coef, lsz, prep, bc0, bc1, error_model=em, empty=0.0)
        build_s = time.perf_counter() - t_build
        return (lambda: rsfai.preproc4(image, **preproc_kw)), apply, build_s

    # csr / csc share build+apply signatures
    if split in ("no", "bbox"):
        if dim == 1:
            build = rsfai.build_bbox_csr_1d if algo == "csr" else rsfai.build_bbox_csc_1d
            data, indices, indptr, bc = build(
                pos0, delta_pos0=(dpos0 if do_split else None), mask=mask,
                bins=npt, allow_pos0_neg=False)
        else:
            build = rsfai.build_bbox_csr_2d if algo == "csr" else rsfai.build_bbox_csc_2d
            data, indices, indptr, bc0, bc1 = build(
                pos0, chi, delta_pos0=(dpos0 if do_split else None),
                delta_pos1=(dchi if do_split else None), mask=mask, bins=npt,
                allow_pos0_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p)
    else:  # full
        if dim == 1:
            build = rsfai.build_full_csr_1d if algo == "csr" else rsfai.build_full_csc_1d
            data, indices, indptr, bc = build(
                corners, mask=mask, bins=npt, allow_pos0_neg=False,
                chi_disc_at_pi=True, pos1_period=TWO_PI)
        else:
            build = rsfai.build_full_csr_2d if algo == "csr" else rsfai.build_full_csc_2d
            data, indices, indptr, bc0, bc1 = build(
                corners, mask=mask, bins=npt, allow_pos0_neg=False,
                chi_disc_at_pi=cdp, pos1_period=p1p)
    if dim == 1:
        integ = rsfai.csr_integrate1d if algo == "csr" else rsfai.csc_integrate1d
        apply = lambda prep: integ(data, indices, indptr, prep, bc, error_model=em, empty=0.0)
    else:
        integ = rsfai.csr_integrate2d if algo == "csr" else rsfai.csc_integrate2d
        apply = lambda prep: integ(data, indices, indptr, prep, bc0, bc1, error_model=em, empty=0.0)
    build_s = time.perf_counter() - t_build
    return (lambda: rsfai.preproc4(image, **preproc_kw)), apply, build_s


def ms(seconds):
    return 1e3 * seconds


def med(ts):
    return statistics.median(ts)


def bench_dim(dim, reps, warmup):
    npttag = "npt_rad=100 x npt_azim=36" if dim == 2 else "npt=1000"
    print(f"\n{'='*92}\n{dim}D  ({npttag})   STEADY-STATE per-frame, median ms\n{'='*92}")
    print(f"{'tuple':<18}{'pyFAI':>9}{'rsFAI':>9}{'speedup':>9}   |  rsFAI breakdown: "
          f"{'preproc':>8}{'+apply':>8}{'  build(1x)':>11}")
    print("-" * 92)
    rows = []
    tuples = TUPLES + (PYFAI_ONLY if dim == 2 else [])
    preproc_med = None  # shared across tuples of this dim; measured once
    for split, algo in tuples:
        d = find_dataset(split, algo, dim)
        name = f"{split}-{algo}"
        if d is None:
            print(f"{name:<18}{'(no dataset)':>40}")
            continue
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]

        py_fn = make_pyfai_fn(d, cfg)
        py_med = med(time_call(py_fn, reps, warmup))

        if (split, algo) in PYFAI_ONLY:
            print(f"{name:<18}{ms(py_med):>9.3f}{'  --':>9}{'(pyFAI only)':>18}")
            rows.append((name, py_med, None, None))
            continue

        preproc_fn, apply_fn, build_s = make_rsfai_fn(d, cfg)
        prep0 = preproc_fn()                       # prebuild prep for apply-only timing
        total_med = med(time_call(lambda: apply_fn(preproc_fn()), reps, warmup))
        apply_med = med(time_call(lambda: apply_fn(prep0), reps, warmup))
        if preproc_med is None:
            preproc_med = med(time_call(preproc_fn, reps, warmup))
        speedup = py_med / total_med
        print(f"{name:<18}{ms(py_med):>9.3f}{ms(total_med):>9.3f}{speedup:>8.2f}x"
              f"   |  {ms(preproc_med):>13.3f}{ms(apply_med):>8.3f}{ms(build_s):>9.1f}")
        rows.append((name, py_med, total_med, speedup))
    return rows


def main():
    omp = os.environ.get("OMP_NUM_THREADS", "(unset->all)")
    ray = os.environ.get("RAYON_NUM_THREADS", "(unset->all)")
    print(f"pyFAI {pyFAI.version} | numpy {np.__version__} | cores={os.cpu_count()}")
    print(f"OMP_NUM_THREADS={omp}  RAYON_NUM_THREADS={ray}")
    print("metric: STEADY-STATE per-frame (geometry+matrix built once, untimed).")
    print("  pyFAI = warm-cached integrate*_ng;  rsFAI = preproc4 + prebuilt-matrix apply.")

    reps, warmup = 60, 8
    r1 = bench_dim(1, reps, warmup)
    r2 = bench_dim(2, reps, warmup)

    # geometric-mean speedup per dim (rsFAI faster when >1)
    def geomean(rows):
        sp = [s for (_, _, _, s) in rows if s]
        return math.exp(sum(math.log(x) for x in sp) / len(sp)) if sp else float("nan")

    print(f"\n{'='*92}")
    print(f"geomean speedup, full per-frame (rsFAI vs pyFAI): "
          f"1D={geomean(r1):.2f}x  2D={geomean(r2):.2f}x   (>1 = rsFAI faster)")


if __name__ == "__main__":
    main()
