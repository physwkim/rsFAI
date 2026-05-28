#!/usr/bin/env python3
"""Steady-state per-frame performance: CPU (cython) vs GPU (opencl), via pyFAI.

Run in the ``daq`` conda env (needs ``pyFAI`` + ``pyopencl`` + a working OpenCL
device):

    /Users/stevek/mamba/envs/daq/bin/python golden/bench_gpu.py

Companion to ``bench_compare.py``. Both backends here are pyFAI's warm-cached
``integrate*_ng`` (the engine — sparse matrix + compiled kernel + normalized
corrections — is built and cached once on the ``ai`` object after warm-up, so a
timed call is one frame: preprocess + integrate, fused). The ONLY thing that
differs between the two columns is the method's implementation backend
(``cython`` on the CPU vs ``opencl`` on the GPU), so the comparison isolates the
single question: is the GPU faster for this per-frame workload?

rsFAI's own OpenCL backend (``rsfai-opencl``) reuses pyFAI's identical ``.cl``
kernels on the same device, so pyFAI's opencl numbers represent the rsFAI GPU
path too; rsFAI does not (yet) expose OpenCL to Python, hence measuring through
pyFAI.

What the numbers showed on an Apple M4 Pro (16-CU GPU, unified memory):

  * At Pilatus1M (~1.0M px) the GPU is ~3x SLOWER (csr/lut ~0.31-0.40x); 2D LUT
    is pathological (~0.03x) because its dense (nbins, lut_size) padded matrix is
    gathered per work-item -- use CSR on the GPU, not LUT-2D.
  * The workload is MEMORY-BOUND (one image pass + a sparse gather, almost no
    arithmetic per byte), so the GPU's per-frame cost is roughly CONSTANT
    (~8-10 ms: kernel dispatch + command-queue + the blocking readback sync,
    nearly independent of pixel count over 1M..4.5M), while the CPU (all cores,
    bandwidth-bound) scales linearly with pixels.
  * Hence a detector-size crossover near ~3M px: at Eiger4M (~4.5M px) the GPU
    pulls ahead (~1.3x). Apple unified memory removes the PCIe copy but not the
    per-dispatch overhead. Levers for the GPU: bigger detectors, batching many
    frames per launch, finer binning.

Correctness is sanity-checked per tuple (max relative intensity difference GPU
vs CPU; csr/lut are bit-exact up to f32, ~1e-7).
"""

import json
import logging
import os
import time
from pathlib import Path

# Quiet the OpenCL compiler chatter and pyFAI's INFO logging so the table is clean.
os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pyFAI  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"

# GPU kernels exist for the sparse engines; csr/lut are the ported, golden-checked
# ones (histogram on the GPU carries atomic-ordering noise, excluded here).
TUPLES = [("no", "csr"), ("bbox", "csr"), ("full", "csr"),
          ("no", "lut"), ("bbox", "lut"), ("full", "lut")]
# Detector-size scaling probe uses one tuple across detectors.
SCALE_TUPLE = ("bbox", "csr")
SCALE_DETECTORS = ("Pilatus1M", "Eiger4M")


def find(split, algo, dim, detector="Pilatus1M"):
    """The canonical cython golden dataset for a tuple (npt1000/100x36, poisson),
    preferring the q_nm^-1 build, excluding OpenCL/range/orientation variants."""
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


def med(fn, reps=60, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def make(d, cfg, impl):
    """A warm-callable for one (dataset, impl): pyFAI integrate*_ng with the method's
    backend forced to `impl`. Returns (ai, call, npixels)."""
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = np.ascontiguousarray(np.load(DATASETS / d / "image.npy").astype(np.float32))
    split, algo = cfg["method"][0], cfg["method"][1]
    common = dict(
        unit=cfg["unit"], method=(split, algo, impl),
        correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"])
    if cfg.get("dim", 1) == 2:
        nr, na = cfg["npt_rad"], cfg["npt_azim"]
        return ai, (lambda: ai.integrate2d_ng(img, nr, na, **common)), img.size
    return ai, (lambda: ai.integrate1d_ng(img, cfg["npt"], **common)), img.size


def max_rel_intensity(cpu_res, gpu_res):
    ic = np.asarray(cpu_res.intensity, dtype=np.float64)
    ig = np.asarray(gpu_res.intensity, dtype=np.float64)
    if ic.shape != ig.shape:
        return float("nan")
    return float(np.max(np.abs(ig - ic) / np.maximum(np.abs(ic), 1e-12)))


def bench_tuples(dim):
    print(f"=== {dim}D  warm-cached pyFAI, CPU cython vs GPU opencl, median ms/frame ===")
    print(f"{'tuple':<12} {'CPU cy':>9} {'GPU ocl':>9} {'speedup':>9} {'maxrel':>10}  note")
    print("-" * 72)
    for split, algo in TUPLES:
        d = find(split, algo, dim)
        if d is None:
            print(f"{split}-{algo}: no dataset")
            continue
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        _, fc, _ = make(d, cfg, "cython")
        rc = fc()
        cpu = med(fc)
        try:
            ai_g, fg, _ = make(d, cfg, "opencl")
            rg = fg()
            gpu = med(fg)
            maxrel = max_rel_intensity(rc, rg)
            used = any("opencl" in str(k).lower() for k in ai_g.engines)
            note = "" if used else "NOT opencl!"
            print(f"{split}-{algo:<7} {cpu:>9.3f} {gpu:>9.3f} {cpu / gpu:>8.2f}x "
                  f"{maxrel:>10.2e}  {note}")
        except Exception as e:  # noqa: BLE001 — report the backend failure, keep going
            print(f"{split}-{algo:<7} {cpu:>9.3f}     GPU failed: {str(e)[:46]}")
    print()


def bench_scaling():
    split, algo = SCALE_TUPLE
    print(f"=== detector-size scaling ({split}-{algo} 1D), CPU vs GPU, median ms/frame ===")
    print(f"{'detector':<12} {'pixels':>10} {'CPU cy':>9} {'GPU ocl':>9} {'speedup':>9}")
    print("-" * 54)
    for det in SCALE_DETECTORS:
        d = find(split, algo, 1, detector=det)
        if d is None:
            print(f"{det}: no dataset")
            continue
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        _, fc, npx = make(d, cfg, "cython")
        cpu = med(fc)
        try:
            _, fg, _ = make(d, cfg, "opencl")
            fg()
            gpu = med(fg)
            print(f"{det:<12} {npx:>10} {cpu:>9.3f} {gpu:>9.3f} {cpu / gpu:>8.2f}x")
        except Exception as e:  # noqa: BLE001
            print(f"{det:<12} {npx:>10} {cpu:>9.3f}     GPU failed: {str(e)[:40]}")
    print()


def main():
    import pyopencl as cl
    devs = [f"{d.name} ({cl.device_type.to_string(d.type)}, {d.max_compute_units} CU)"
            for p in cl.get_platforms() for d in p.get_devices()]
    print(f"pyFAI {pyFAI.version} | OpenCL devices: {', '.join(devs) or 'none'}")
    print("metric: STEADY-STATE per-frame (engine cached, per call = preproc+integrate)\n")
    bench_tuples(1)
    bench_tuples(2)
    bench_scaling()


if __name__ == "__main__":
    main()
