#!/usr/bin/env python3
"""Binning scaling, four-way: pyFAI-CPU vs pyFAI-GPU vs rsFAI-CPU, 1D and 2D.

Run in the ``daq`` conda env (needs ``pyFAI`` + ``pyopencl`` + a working OpenCL
device + the maturin-built ``rsfai``)::

    python golden/bench_npt.py                       # 1D 1000,5000 + 2D 1000x360,5000x360
    python golden/bench_npt.py 1000 5000             # 1D only, these npt
    python golden/bench_npt.py 1000x360 5000x360     # 2D only, these (npt_rad x npt_azim)
    python golden/bench_npt.py 2000 2000x720         # mix: one 1D level, one 2D level

Bare integers are 1D ``npt``; ``<nr>x<na>`` tokens are 2D ``(npt_rad, npt_azim)``.
Companion to ``bench_compare.py`` (CPU pyFAI-vs-rsFAI, full tuple matrix incl.
histogram/csc) and ``bench_gpu.py`` (CPU-vs-GPU + detector-size scaling). This one
varies the bin count to answer "does finer binning shift the CPU/GPU/rsFAI balance
at a fixed detector size?".

Metric = STEADY-STATE per-frame (matrix/engine built once, untimed; only the
per-frame preproc+apply timed):

  * pyFAI CPU = warm-cached ``integrate{1,2}d_ng``, method ``(split, algo, 'cython')``.
  * pyFAI GPU = same, ``'opencl'`` (Apple M4 Pro). rsFAI reuses pyFAI's identical
    ``.cl`` kernels on the same device and does not expose OpenCL to Python, so
    pyFAI-GPU represents the rsFAI GPU path too.
  * rsFAI CPU = ``preproc4`` + cached ``Engine.integrate{1,2}d`` (CSR/LUT built at
    ``bins=npt`` from the dataset's dumped positions, held Rust-side, so there is
    no per-frame matrix copy).

CORRECTNESS (``maxrel`` column) = max relative intensity diff rsFAI vs pyFAI-CPU
on non-empty bins. It is 0 (bit-identical) at every binning because the rebuilt
matrix reproduces pyFAI's bit-for-bit and ``preproc4`` is called with pyFAI's
normalization convention:

  * ``apply_normalization=False`` -- pyFAI's ``integrate*_ng`` DEFERS the
    correction division to the integrator (signal stays raw, norm holds the
    correction product, ``intensity = Σsignal·c / Σnorm·c``). The default True
    pre-divides signal and yields an unweighted mean instead of pyFAI's
    solid-angle-weighted mean -- a ~1e-3 mismatch that only shows when the
    correction varies within a bin.
  * ``polarization.npy`` is passed iff the dataset's ``polarization_factor`` is set.
  * 2D needs the manifest's ``pos1_period`` (360 -- chi in degrees, not radians)
    and ``chi_disc_at_pi``.

What the numbers showed on an Apple M4 Pro (Pilatus1M, 1.02M px), 1000 -> 5000
radial bins: the ranking stays rsFAI-CPU > pyFAI-CPU > GPU. The GPU gap narrows
only because pyFAI-CPU slows with more bins while the GPU stays ~flat (memory-bound:
one image pass dominates) -- no crossover from binning. Binning is a weaker GPU lever
than DETECTOR SIZE (see bench_gpu.py: Eiger4M 4.5M px is where the GPU wins,
crossover ~2M px by pixel count). 2D LUT is only pathological at COARSE binning
(few azimuthal bins -> huge ``lut_size``); it is fine once bins are sparse.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Quiet the OpenCL compiler chatter and pyFAI's INFO logging so the table is clean.
os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pyFAI  # noqa: E402
import rsfai  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
TWO_PI = 2.0 * np.pi

# csr/lut are the ported, golden-checked sparse engines; histogram has no reusable
# matrix and CSC is serial-by-design, so they are out of this matrix.
TUPLES = [("no", "csr"), ("bbox", "csr"), ("full", "csr"),
          ("no", "lut"), ("bbox", "lut"), ("full", "lut")]


def npy(d, name):
    return np.ascontiguousarray(np.load(DATASETS / d / name))


def find(split, algo, dim, detector="Pilatus1M"):
    """The canonical cython golden dataset for a tuple (npt1000/npt100x36, poisson),
    preferring q_nm^-1, excluding OpenCL/range/orientation variants. Only the
    geometry/image/positions are reused; the bin count is overridden per run."""
    npttag = "npt100x36" if dim == 2 else "npt1000"
    cands = []
    for p in DATASETS.iterdir():
        n = p.name
        if not n.startswith(f"{detector}__{split}-{algo}-cython__"):
            continue
        if npttag not in n or "errpoisson" not in n:
            continue
        if any(x in n for x in ("razim", "rrad", "orient", "opencl")):
            continue
        if not (p / "manifest.json").exists():
            continue
        if json.load(open(p / "manifest.json"))["config"].get("dim", 1) != dim:
            continue
        cands.append(n)
    cands.sort(key=lambda n: (0 if "q_nmm1" in n else 1, n))
    return cands[0] if cands else None


def med(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def pyfai_call(d, cfg, impl, bins, dim):
    """Warm-callable pyFAI integrate{1,2}d_ng for one (dataset, backend) at these bins."""
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = np.ascontiguousarray(npy(d, "image.npy").astype(np.float32))
    split, algo = cfg["method"][0], cfg["method"][1]
    common = dict(unit=cfg["unit"], method=(split, algo, impl),
                  correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
                  polarization_factor=cfg["polarization_factor"],
                  normalization_factor=cfg["normalization_factor"])
    if dim == 2:
        nr, na = bins
        return lambda: ai.integrate2d_ng(img, nr, na, **common)
    return lambda: ai.integrate1d_ng(img, bins, **common)


def make_prep_fn(d, cfg):
    """Single source of truth for rsFAI preproc -- mirrors exactly the corrections
    pyFAI applied (solidangle iff correctSolidAngle; polarization iff
    polarization_factor set; poisson per error_model; dummy masking) and uses
    pyFAI's DEFERRED-normalization convention (apply_normalization=False).
    Returns (() -> prep (npix,4), error_model_code)."""
    image = np.ascontiguousarray(npy(d, "image.npy").astype(np.float32).reshape(-1))
    mask = npy(d, "mask.npy").reshape(-1).astype(np.int8)
    sa = npy(d, "solidangle.npy").astype(np.float32).reshape(-1) \
        if cfg["correct_solid_angle"] and (DATASETS / d / "solidangle.npy").exists() else None
    pol = npy(d, "polarization.npy").astype(np.float32).reshape(-1) \
        if cfg["polarization_factor"] is not None and (DATASETS / d / "polarization.npy").exists() else None
    dummy = cfg["dummy"]
    em = cfg["error_model_code"]
    kw = dict(solidangle=sa, polarization=pol, mask=mask,
              normalization_factor=np.float32(cfg["normalization_factor"]),
              poissonian=(em == 2), check_dummy=(dummy is not None),
              dummy=float(dummy if dummy is not None else 0.0),
              delta_dummy=float(cfg["delta_dummy"] if cfg["delta_dummy"] is not None else 0.0),
              apply_normalization=False)
    return (lambda: rsfai.preproc4(image, **kw)), em


def build_engine(d, cfg, split, algo, bins, dim):
    """Build the rsFAI CSR/LUT matrix at `bins` from the dataset's dumped positions
    and wrap it in a cached Engine (matrix copied once, here)."""
    mask = npy(d, "mask.npy").reshape(-1).astype(np.int8)
    pos0 = npy(d, "pos0_center_unscaled.npy").reshape(-1)
    corners = npy(d, "corners.npy").astype(np.float64).reshape(-1)
    if dim == 1:
        dpos0 = npy(d, "pos0_delta.npy").reshape(-1) if split == "bbox" else None
        if algo == "lut":
            if split in ("no", "bbox"):
                idx, coef, lsz, bc = rsfai.build_bbox_lut_1d(pos0, delta_pos0=dpos0, mask=mask, bins=bins, allow_pos0_neg=False)
            else:
                idx, coef, lsz, bc = rsfai.build_full_lut_1d(corners, mask=mask, bins=bins, allow_pos0_neg=False, chi_disc_at_pi=True, pos1_period=TWO_PI)
            return rsfai.Engine.from_lut_1d(idx, coef, lsz, bc)
        if split in ("no", "bbox"):
            data, indices, indptr, bc = rsfai.build_bbox_csr_1d(pos0, delta_pos0=dpos0, mask=mask, bins=bins, allow_pos0_neg=False)
        else:
            data, indices, indptr, bc = rsfai.build_full_csr_1d(corners, mask=mask, bins=bins, allow_pos0_neg=False, chi_disc_at_pi=True, pos1_period=TWO_PI)
        return rsfai.Engine.from_csr_1d(data, indices, indptr, bc)

    # dim == 2: pos1 = chi (degrees; manifest pos1_period=360), split needs both deltas
    chi = npy(d, "chi_center.npy").reshape(-1)
    dpos0 = npy(d, "pos0_delta.npy").reshape(-1) if split == "bbox" else None
    dchi = npy(d, "chi_delta.npy").reshape(-1) if split == "bbox" else None
    cdp = cfg.get("chi_disc_at_pi", True)
    p1p = cfg.get("pos1_period", TWO_PI)
    if algo == "lut":
        if split in ("no", "bbox"):
            idx, coef, lsz, bc0, bc1 = rsfai.build_bbox_lut_2d(pos0, chi, delta_pos0=dpos0, delta_pos1=dchi, mask=mask, bins=bins, allow_pos0_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p)
        else:
            idx, coef, lsz, bc0, bc1 = rsfai.build_full_lut_2d(corners, mask=mask, bins=bins, allow_pos0_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p)
        return rsfai.Engine.from_lut_2d(idx, coef, lsz, bc0, bc1)
    if split in ("no", "bbox"):
        data, indices, indptr, bc0, bc1 = rsfai.build_bbox_csr_2d(pos0, chi, delta_pos0=dpos0, delta_pos1=dchi, mask=mask, bins=bins, allow_pos0_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p)
    else:
        data, indices, indptr, bc0, bc1 = rsfai.build_full_csr_2d(corners, mask=mask, bins=bins, allow_pos0_neg=False, chi_disc_at_pi=cdp, pos1_period=p1p)
    return rsfai.Engine.from_csr_2d(data, indices, indptr, bc0, bc1)


def bench(bins, dim, reps, warmup):
    label = f"{bins[0]}x{bins[1]} (rad x azim)" if dim == 2 else f"npt={bins}"
    print(f"=== {dim}D {label}, Pilatus1M ~1.02M px, warm steady-state per-frame, median ms ===")
    print(f"{'tuple':<11} {'pyFAI-CPU':>10} {'pyFAI-GPU':>10} {'rsFAI-CPU':>10} {'rs-apply':>9} "
          f"{'rs/pyCPU':>9} {'GPU/pyCPU':>10} {'maxrel':>9}")
    print("-" * 92)
    for split, algo in TUPLES:
        d = find(split, algo, dim)
        if d is None:
            print(f"{split}-{algo}: no dataset")
            continue
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        fc = pyfai_call(d, cfg, "cython", bins, dim)
        rc = fc()
        pcpu = med(fc, reps, warmup)
        try:
            fg = pyfai_call(d, cfg, "opencl", bins, dim)
            fg()
            pgpu = med(fg, reps, warmup)
        except Exception:  # noqa: BLE001 -- GPU backend may be unavailable; keep CPU columns
            pgpu = float("nan")
        eng = build_engine(d, cfg, split, algo, bins, dim)
        make_prep, em = make_prep_fn(d, cfg)
        prep0 = make_prep()
        apply = eng.integrate2d if dim == 2 else eng.integrate1d
        rtot = med(lambda: apply(make_prep(), error_model=em, empty=0.0), reps, warmup)
        rapp = med(lambda: apply(prep0, error_model=em, empty=0.0), reps, warmup)
        out = apply(prep0, error_model=em, empty=0.0)
        ip = np.asarray(rc.intensity, dtype=np.float64)
        ir = np.asarray(out["intensity"], dtype=np.float64).reshape(ip.shape)  # 2D: (npt_azim, npt_rad)
        nz = np.abs(ip) > 1e-12
        maxrel = float(np.max(np.abs(ir - ip)[nz] / np.abs(ip)[nz])) if nz.any() else float("nan")
        sp = pcpu / rtot
        gsp = pcpu / pgpu if pgpu == pgpu else float("nan")
        print(f"{split}-{algo:<6} {pcpu:>10.3f} {pgpu:>10.3f} {rtot:>10.3f} {rapp:>9.3f} "
              f"{sp:>8.2f}x {gsp:>9.2f}x {maxrel:>9.1e}")
    print()


def parse_args(argv):
    """Bare ints -> 1D npt; '<nr>x<na>' -> 2D (rad, azim). No args -> both defaults."""
    npts_1d, levels_2d = [], []
    for a in argv:
        if "x" in a.lower():
            nr, na = a.lower().split("x")
            levels_2d.append((int(nr), int(na)))
        else:
            npts_1d.append(int(a))
    if not npts_1d and not levels_2d:
        return [1000, 5000], [(1000, 360), (5000, 360)]
    return npts_1d, levels_2d


def main():
    npts_1d, levels_2d = parse_args(sys.argv[1:])
    import pyopencl as cl
    devs = [f"{x.name} ({cl.device_type.to_string(x.type)})" for p in cl.get_platforms() for x in p.get_devices()]
    print(f"pyFAI {pyFAI.version} | GPU: {', '.join(devs) or 'none'}")
    print("rsFAI-GPU == pyFAI-GPU (same .cl kernels; rsFAI OpenCL not exposed to Python)\n")
    for npt in npts_1d:
        bench(npt, 1, reps=50, warmup=10)
    for bins in levels_2d:  # 2D apply is heavier (up to ~50 ms/frame at 5000x360); fewer reps
        bench(bins, 2, reps=20, warmup=5)


if __name__ == "__main__":
    main()
