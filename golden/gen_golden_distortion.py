#!/usr/bin/env python
"""Golden generator for distortion + spline (rsfai-distortion crate).

Run single-thread in the daq env (pyFAI 2026.5.0, no-FMA source build):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_distortion.py

Three independently-gated layers:

  1. bisplev  -- pyFAI _bispev.bisplev(x, y, tck) on the halfccd X-displacement
     and Y-displacement tensors over a small evaluation grid. Gates the de
     Boor-Cox + Kahan tensor eval bit-for-bit. Dumps the knots/coeffs (small,
     committed) + eval axes + the surface values.

  2. spline2array -- Spline.spline2array() over the *full* detector grid (the
     displacement maps the FReLoN detector consumes). Large arrays, gitignored.

  3. Distortion LUT + correct -- a binned FReLoN(halfccd.spline) detector
     (binning 5x8 -> 205x256, real spline-driven distortion, manageable size).
     Dumps _distortion.calc_pos's `pos` 4D array as a Tier-A INPUT, the CSR
     (data/indices/indptr) as golden, a deterministic raw image as input, and
     the corrected image as golden. Large arrays gitignored, regenerated here.

The halfccd.spline fixture is committed under datasets_distortion/ (1.2 kB).
Everything is float32 per pyFAI's distortion dtype contract; `pos`, CSR data,
and the corrected image must reproduce bit-for-bit.
"""

import json
import os

import numpy as np

from pyFAI import detectors, distortion
from pyFAI.ext import _distortion, _bispev
from pyFAI.spline import Spline

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_distortion")
SPLINE_FILE = os.path.join(OUTDIR, "halfccd.spline")

# Binning shrinks the FReLoN's (1025, 2048) to (205, 256): a real
# spline-distorted detector small enough for a committable golden.
BINNING = (5, 8)


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name), arr)
    return list(arr.shape), str(arr.dtype)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    assert os.path.isfile(SPLINE_FILE), f"missing fixture {SPLINE_FILE}"

    manifest = {
        "dataset": "distortion",
        "source": "gen_golden_distortion.py",
        "spline_file": "halfccd.spline",
        "pyfai_version": __import__("pyFAI").version,
        "numpy_version": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }

    # ---- Layer 1 + parser fixture: parse the spline, dump knots/coeffs ----
    sp = Spline(SPLINE_FILE)
    manifest["spline"] = {
        "xmin": sp.xmin, "ymin": sp.ymin, "xmax": sp.xmax, "ymax": sp.ymax,
        "grid": sp.grid, "pixel_size": list(sp.pixelSize), "order": sp.splineOrder,
    }
    # Small enough to commit (out_* naming): knots + coeffs for both tensors.
    sp_arrays = {}
    for tag, kx, ky, c in [
        ("x", sp.xSplineKnotsX, sp.xSplineKnotsY, sp.xSplineCoeff),
        ("y", sp.ySplineKnotsX, sp.ySplineKnotsY, sp.ySplineCoeff),
    ]:
        sp_arrays[f"out_spline_{tag}_knotsx"] = save(f"out_spline_{tag}_knotsx.npy", kx)
        sp_arrays[f"out_spline_{tag}_knotsy"] = save(f"out_spline_{tag}_knotsy.npy", ky)
        sp_arrays[f"out_spline_{tag}_coeff"] = save(f"out_spline_{tag}_coeff.npy", c)
    manifest["spline_arrays"] = sp_arrays

    # ---- Layer 1: bisplev over a small evaluation grid ----
    # A small, asymmetric grid so we exercise both axes (5 x 4) and a mid-range
    # subset (the points lie inside [xmin, xmax] / [ymin, ymax]).
    bx = np.arange(0.0, 50.0, 11.0)   # 5 points: 0,11,22,33,44
    by = np.arange(3.0, 40.0, 9.0)    # 5 points: 3,12,21,30,39
    save("out_bisplev_x.npy", bx.astype(np.float32))
    save("out_bisplev_y.npy", by.astype(np.float32))
    zx = _bispev.bisplev(bx, by, [sp.xSplineKnotsX, sp.xSplineKnotsY, sp.xSplineCoeff, 3, 3])
    zy = _bispev.bisplev(bx, by, [sp.ySplineKnotsX, sp.ySplineKnotsY, sp.ySplineCoeff, 3, 3])
    # bisplev returns shape (len(x), len(y)) for >1 points; record both.
    manifest["bisplev"] = {
        "x": save("out_bisplev_zx.npy", np.asarray(zx, dtype=np.float32)),
        "y": save("out_bisplev_zy.npy", np.asarray(zy, dtype=np.float32)),
    }

    # ---- Layer 2: spline2array over the full grid (gitignored, big) ----
    xdisp, ydisp = sp.spline2array()
    manifest["spline2array"] = {
        "xDispArray": save("spline2array_xdisp.npy", np.asarray(xdisp, dtype=np.float32)),
        "yDispArray": save("spline2array_ydisp.npy", np.asarray(ydisp, dtype=np.float32)),
    }

    # ---- Layer 3: Distortion LUT (CSR) + correct ----
    det = detectors.FReLoN(SPLINE_FILE)
    det.binning = BINNING
    shape = det.shape
    corners = det.get_pixel_corners()  # (nrow, ncol, 4, 3) float32
    pos, delta1, delta2, shape_out, offset = _distortion.calc_pos(
        corners, det.pixel1, det.pixel2, shape
    )
    manifest["distortion"] = {
        "binning": list(BINNING),
        "shape_in": list(shape),
        "shape_out": list(shape_out),
        "delta": [int(delta1), int(delta2)],
        "offset": [float(offset[0]), float(offset[1])],
        "pixel1": float(det.pixel1),
        "pixel2": float(det.pixel2),
        "empty": 0.0,
    }
    # Tier-A INPUT: the pos 4D array (so the Rust LUT build is fed identical
    # corner positions). Gitignored (regenerable, ~1.7 MB).
    manifest["distortion"]["pos"] = save("dist_pos.npy", np.asarray(pos, dtype=np.float32))
    # Also dump the raw corner array as a secondary input (lets the verifier
    # additionally exercise the Rust calc_pos against pyFAI's pos).
    manifest["distortion"]["corners"] = save("dist_corners.npy", np.asarray(corners, dtype=np.float32))

    dis = distortion.Distortion(det, method="csr")
    dis.reset(prepare=True)
    data, indices, indptr = dis.lut
    manifest["distortion"]["csr"] = {
        "data": save("dist_csr_data.npy", np.asarray(data, dtype=np.float32)),
        "indices": save("dist_csr_indices.npy", np.asarray(indices, dtype=np.int32)),
        "indptr": save("dist_csr_indptr.npy", np.asarray(indptr, dtype=np.int32)),
    }

    # Deterministic raw image (input) + corrected image (golden).
    rng = np.random.default_rng(2026)
    img = (rng.random(shape) * 1000.0).astype(np.float32)
    manifest["distortion"]["image"] = save("dist_image.npy", img)
    cor = dis.correct(img)
    manifest["distortion"]["corrected"] = save(
        "out_dist_corrected.npy", np.asarray(cor, dtype=np.float32)
    )

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote distortion golden to {OUTDIR}")
    print(f"  spline: region [{sp.xmin},{sp.xmax}] x [{sp.ymin},{sp.ymax}]")
    print(f"  bisplev grid: {bx.size} x {by.size}, zx{np.asarray(zx).shape} zy{np.asarray(zy).shape}")
    print(f"  detector binned {BINNING} -> {shape}, out {shape_out}, delta ({delta1},{delta2})")
    print(f"  CSR: data {data.shape} indices {indices.shape} indptr {indptr.shape}")
    print(f"  corrected sum {cor.sum():.6g}")


if __name__ == "__main__":
    main()
