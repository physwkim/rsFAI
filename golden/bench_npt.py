#!/usr/bin/env python3
"""npt (binning) scaling, four-way: pyFAI-CPU vs pyFAI-GPU vs rsFAI-CPU, 1D.

Run in the ``daq`` conda env (needs ``pyFAI`` + ``pyopencl`` + a working OpenCL
device + the maturin-built ``rsfai``)::

    /Users/stevek/mamba/envs/daq/bin/python golden/bench_npt.py            # 1000 5000
    /Users/stevek/mamba/envs/daq/bin/python golden/bench_npt.py 1000 5000 20000

Companion to ``bench_compare.py`` (CPU pyFAI-vs-rsFAI) and ``bench_gpu.py``
(pyFAI CPU-vs-GPU). This one varies the bin count to answer "does finer binning
shift the CPU/GPU/rsFAI balance at a fixed detector size?".

Metric = STEADY-STATE per-frame (matrix/engine built once, untimed; only the
per-frame preproc+apply timed):

  * pyFAI CPU = warm-cached ``integrate1d_ng``, method ``(split, algo, 'cython')``.
  * pyFAI GPU = same, ``'opencl'`` (Apple M4 Pro). rsFAI reuses pyFAI's identical
    ``.cl`` kernels on the same device and does not expose OpenCL to Python, so
    pyFAI-GPU represents the rsFAI GPU path too.
  * rsFAI CPU = ``preproc4`` + cached ``Engine.integrate1d`` (CSR/LUT built at
    ``bins=npt`` from the dataset's dumped unscaled positions, held Rust-side, so
    there is no per-frame matrix copy).

CORRECTNESS (``maxrel`` column) = max relative intensity diff rsFAI vs pyFAI-CPU
on non-empty bins. It is 0 (bit-identical) at every npt because the rebuilt CSR
matrix reproduces pyFAI's matrix bit-for-bit (scale cancels in the overlap
fractions) and ``preproc4`` is called with pyFAI's normalization convention:

  * ``apply_normalization=False`` -- pyFAI's ``integrate*_ng`` DEFERS the
    correction division to the integrator (signal stays raw, norm holds the
    correction product, ``intensity = Σsignal·c / Σnorm·c``). The default True
    pre-divides signal and yields an unweighted mean instead of pyFAI's
    solid-angle-weighted mean -- a ~1e-3 mismatch that only shows when the
    correction varies within a bin.
  * ``polarization.npy`` is passed iff the dataset's ``polarization_factor`` is
    set (else pyFAI applies a correction this path would silently omit).

What the numbers showed on an Apple M4 Pro (Pilatus1M, 1.02M px), 1000 -> 5000:
the ranking stays rsFAI-CPU > pyFAI-CPU > GPU. The GPU gap narrows (~0.55x ->
~0.78x) only because pyFAI-CPU slows with 5x bins while the GPU stays ~flat
(memory-bound: one image pass dominates, 5x output bins is negligible) -- no
crossover. rsFAI-CPU's lead over pyFAI-CPU grows on split tuples (1.5x -> 1.8x;
its per-bin rayon reduction absorbs more bins better). Binning is a weaker GPU
lever than DETECTOR SIZE (see bench_gpu.py: Eiger4M 4.5M px is where the GPU
wins, crossover ~2M px by pixel count).
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


def find(split, algo, detector="Pilatus1M"):
    """The canonical cython golden dataset for a 1D tuple (npt1000, poisson),
    preferring q_nm^-1, excluding OpenCL/range/orientation variants. Only the
    geometry/image/positions are reused; the npt is overridden per run."""
    cands = []
    for p in DATASETS.iterdir():
        n = p.name
        if not n.startswith(f"{detector}__{split}-{algo}-cython__"):
            continue
        if "npt1000" not in n or "errpoisson" not in n:
            continue
        if any(x in n for x in ("razim", "rrad", "orient", "opencl")):
            continue
        if not (p / "manifest.json").exists():
            continue
        if json.load(open(p / "manifest.json"))["config"].get("dim", 1) != 1:
            continue
        cands.append(n)
    cands.sort(key=lambda n: (0 if "q_nmm1" in n else 1, n))
    return cands[0] if cands else None


def med(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def pyfai_call(d, cfg, impl, npt):
    """Warm-callable pyFAI integrate1d_ng for one (dataset, backend) at this npt."""
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = np.ascontiguousarray(npy(d, "image.npy").astype(np.float32))
    split, algo = cfg["method"][0], cfg["method"][1]
    common = dict(unit=cfg["unit"], method=(split, algo, impl),
                  correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
                  polarization_factor=cfg["polarization_factor"],
                  normalization_factor=cfg["normalization_factor"])
    return lambda: ai.integrate1d_ng(img, npt, **common)


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


def build_engine(d, split, algo, npt):
    """Build the rsFAI CSR/LUT matrix at bins=npt from the dataset's dumped
    positions and wrap it in a cached Engine (matrix copied once, here)."""
    mask = npy(d, "mask.npy").reshape(-1).astype(np.int8)
    pos0 = npy(d, "pos0_center_unscaled.npy").reshape(-1)
    dpos0 = npy(d, "pos0_delta.npy").reshape(-1) if split == "bbox" else None
    corners = npy(d, "corners.npy").astype(np.float64).reshape(-1) if split == "full" else None
    if algo == "lut":
        if split in ("no", "bbox"):
            idx, coef, lsz, bc = rsfai.build_bbox_lut_1d(pos0, delta_pos0=dpos0, mask=mask, bins=npt, allow_pos0_neg=False)
        else:
            idx, coef, lsz, bc = rsfai.build_full_lut_1d(corners, mask=mask, bins=npt, allow_pos0_neg=False, chi_disc_at_pi=True, pos1_period=TWO_PI)
        return rsfai.Engine.from_lut_1d(idx, coef, lsz, bc)
    if split in ("no", "bbox"):
        data, indices, indptr, bc = rsfai.build_bbox_csr_1d(pos0, delta_pos0=dpos0, mask=mask, bins=npt, allow_pos0_neg=False)
    else:
        data, indices, indptr, bc = rsfai.build_full_csr_1d(corners, mask=mask, bins=npt, allow_pos0_neg=False, chi_disc_at_pi=True, pos1_period=TWO_PI)
    return rsfai.Engine.from_csr_1d(data, indices, indptr, bc)


def bench(npt):
    print(f"=== npt={npt} 1D, Pilatus1M ~1.02M px, warm steady-state per-frame, median ms ===")
    print(f"{'tuple':<11} {'pyFAI-CPU':>10} {'pyFAI-GPU':>10} {'rsFAI-CPU':>10} {'rs-apply':>9} "
          f"{'rs/pyCPU':>9} {'GPU/pyCPU':>10} {'maxrel':>9}")
    print("-" * 92)
    for split, algo in TUPLES:
        d = find(split, algo)
        if d is None:
            print(f"{split}-{algo}: no dataset")
            continue
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        fc = pyfai_call(d, cfg, "cython", npt)
        rc = fc()
        pcpu = med(fc)
        try:
            fg = pyfai_call(d, cfg, "opencl", npt)
            fg()
            pgpu = med(fg)
        except Exception:  # noqa: BLE001 -- GPU backend may be unavailable; keep CPU columns
            pgpu = float("nan")
        eng = build_engine(d, split, algo, npt)
        make_prep, em = make_prep_fn(d, cfg)
        prep0 = make_prep()
        rtot = med(lambda: eng.integrate1d(make_prep(), error_model=em, empty=0.0))
        rapp = med(lambda: eng.integrate1d(prep0, error_model=em, empty=0.0))
        out = eng.integrate1d(prep0, error_model=em, empty=0.0)
        ir = np.asarray(out["intensity"], dtype=np.float64)
        ip = np.asarray(rc.intensity, dtype=np.float64)
        nz = np.abs(ip) > 1e-12
        maxrel = float(np.max(np.abs(ir - ip)[nz] / np.abs(ip)[nz])) if nz.any() else float("nan")
        sp = pcpu / rtot
        gsp = pcpu / pgpu if pgpu == pgpu else float("nan")
        print(f"{split}-{algo:<6} {pcpu:>10.3f} {pgpu:>10.3f} {rtot:>10.3f} {rapp:>9.3f} "
              f"{sp:>8.2f}x {gsp:>9.2f}x {maxrel:>9.1e}")
    print()


def main():
    npts = [int(x) for x in sys.argv[1:]] or [1000, 5000]
    import pyopencl as cl
    devs = [f"{x.name} ({cl.device_type.to_string(x.type)})" for p in cl.get_platforms() for x in p.get_devices()]
    print(f"pyFAI {pyFAI.version} | GPU: {', '.join(devs) or 'none'}")
    print("rsFAI-GPU == pyFAI-GPU (same .cl kernels; rsFAI OpenCL not exposed to Python)\n")
    for npt in npts:
        bench(npt)


if __name__ == "__main__":
    main()
