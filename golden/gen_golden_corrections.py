#!/usr/bin/env python3
"""Golden generator for the **corrections** path of ``integrate1d/2d_with`` —
the entry ``MultiGeometry`` drives per geometry: a user ``mask`` (which
*replaces* the detector mask), ``dark``, ``flat``, an explicit ``variance``, and
a non-unit ``normalization_factor`` (the solid-angle-scaled per-geometry monitor,
computed in f64 then cast to f32 at preproc).

The committed ``golden/datasets`` only exercise ``mask=None``, no dark/flat,
``normalization_factor=1.0`` — so the new ``Corrections`` plumbing in
``rsfai::AzimuthalIntegrator::integrate1d_with`` / ``integrate2d_with`` has no
existing gate. This writes ``golden/datasets_corrections/<key>/`` mirroring
``gen_golden.py``'s layout, consumed by ``crates/rsfai/tests/corrections_golden.rs``.

Run in the ``daq`` conda env, single-thread (the bit-exact gate):

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/gen_golden_corrections.py

Method is ``("bbox","csr","cython")`` (deterministic ⇒ bit-exact) for every
config; the two error models exercise the two preproc variance routes:

  * ``poisson``  — for a non-``manage_variance`` engine (CSR/LUT) pyFAI's
    ``_normalize_error_model_variance`` *precomputes* the per-pixel variance as
    ``max(data,1) + max(dark,0)`` and feeds it as ``VARIANCE`` (preproc poisson
    itself, preproc.pyx:309, is ``max(data,1)`` with no dark term). Whether rsFAI
    reproduces that dark augmentation is exactly what this dataset gates.
  * ``variance`` — an explicit per-pixel variance array, used verbatim by preproc
    (no poisson precompute); isolates the dark-signal subtraction + flat + mask +
    scaled-normalization carry.

``manage_variance`` for the chosen method is recorded in each manifest so the
Rust comparator (and a human reading a failure) can see which route pyFAI took.
"""

import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyFAI  # noqa: E402
from pyFAI import method_registry, units  # noqa: E402
from pyFAI.containers import ErrorModel  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "datasets_corrections"
# A committed Pilatus1M frame (int32) + its PONI — geometry/image source only;
# this generator does not read that dataset's golden.
SRC = HERE / "datasets" / "Pilatus1M__bbox-csc-cython__q_nmm1__npt1000__errazimuthal"

# Every method is a deterministic (serial) cython engine ⇒ every field bit-exact.
# The poisson configs probe both sides of the `manage_variance` boundary so the
# dark-augmented-variance precompute is gated for the engines that need it AND
# confirmed absent for the engines that manage variance themselves:
#   (dim, split, algo, error_model, with_explicit_variance)
CONFIGS = [
    (1, "bbox", "csr", "poisson", False),   # 1D: manage_variance=True
    (1, "bbox", "csr", "variance", True),    # explicit variance path
    (2, "bbox", "csr", "poisson", False),    # 2D manage_variance=False (dark term)
    (2, "bbox", "csr", "variance", True),
    (2, "no", "csr", "poisson", False),      # 2D manage_variance=False
    (2, "bbox", "lut", "poisson", False),    # 2D manage_variance=False
    (2, "full", "csr", "poisson", False),    # 2D manage_variance=True  (no dark term)
    (2, "no", "lut", "poisson", False),      # 2D manage_variance=True  (boundary vs bbox-lut)
    # --- histogram engines with corrections (the MultiGeometry default method,
    #     never covered by the CSR/LUT configs above) -----------------------
    (1, "full", "histogram", "poisson", False),
    (1, "bbox", "histogram", "poisson", False),
    (2, "full", "histogram", "poisson", False),
    (2, "bbox", "histogram", "poisson", False),
    # --- azimuthal error model with corrections (Welford variance per bin) ---
    (1, "bbox", "csr", "azimuthal", False),
    (2, "bbox", "csr", "azimuthal", False),
    (2, "full", "csr", "azimuthal", False),
    (2, "full", "histogram", "azimuthal", False),
]


def _save(meta, out_dir, name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(out_dir / f"{name}.npy", arr)
    meta[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}


def _manage_variance(method, dim):
    base = method_registry.Method(dim, *method, None)
    found = method_registry.IntegrationMethod.select_method(method=base)
    return bool(found[0].manage_variance) if found else None


def main():
    print(f"pyFAI {pyFAI.version}, numpy {np.__version__}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    ai = pyFAI.load(str(SRC / "geometry.poni"))
    data = np.ascontiguousarray(np.load(SRC / "image.npy"))
    shape = data.shape
    data_f32 = data.astype(np.float32)

    # --- Corrections (the paths the committed golden never covers) -----------
    # dark: a non-trivial constant so max(dark,0) shifts the poisson variance
    # visibly; flat: != 1 so it actually scales the denominator; user mask: a
    # rectangle that REPLACES the detector mask (create_mask semantics); explicit
    # variance: an arbitrary positive per-pixel array.
    dark = np.full(shape, 2.0, dtype=np.float32)
    flat = np.full(shape, 1.05, dtype=np.float32)
    user_mask = np.zeros(shape, dtype=np.int8)
    user_mask[100:140, 200:260] = 1
    variance = (np.maximum(data_f32, 1.0) + 3.0).astype(np.float32)

    # Solid-angle-scaled per-geometry monitor, exactly as MultiGeometry forms it:
    # monitor *= pixel1*pixel2/dist**2 in f64. Carried as f64 through Corrections,
    # cast to f32 at preproc (matching pyFAI's `floating normalization_factor`).
    monitor = 2.7
    norm_factor = float(monitor * ai.detector.pixel1 * ai.detector.pixel2 / ai.dist ** 2)

    geom = {k: float(getattr(ai, k)) for k in
            ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength")}
    det = {"name": ai.detector.name, "pixel1": float(ai.detector.pixel1),
           "pixel2": float(ai.detector.pixel2), "shape": list(shape),
           "orientation": int(ai.detector.orientation)}
    dummy_v, delta_dummy_v = ai._normalize_dummies(None, None, data)

    unit = "q_nm^-1"
    npt, npt_rad, npt_azim = 1000, 1000, 360

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    for dim, split, algo, error_model, with_var in CONFIGS:
        method = (split, algo, "cython")
        var_arg = variance if with_var else None
        em = ErrorModel.parse(error_model)
        key = f"Pilatus1M__{'-'.join(method)}__{unit.replace('^','').replace('-','m').replace('/','_')}" \
              f"__dim{dim}__err{error_model}__corr"
        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True)
        arrays = {}

        # ---- Inputs (everything the Rust side feeds back in) ----------------
        _save(arrays, out_dir, "image", data)
        _save(arrays, out_dir, "dark", dark)
        _save(arrays, out_dir, "flat", flat)
        _save(arrays, out_dir, "user_mask", user_mask)
        if with_var:
            _save(arrays, out_dir, "variance", variance)
        shutil.copyfile(SRC / "geometry.poni", out_dir / "geometry.poni")

        common = dict(unit=unit, method=method, correctSolidAngle=True,
                      error_model=error_model, polarization_factor=None,
                      normalization_factor=norm_factor,
                      mask=user_mask, dark=dark, flat=flat, variance=var_arg)
        if dim == 2:
            res = ai.integrate2d_ng(data, npt_rad, npt_azim, **common)
            out_fields = ("radial", "azimuthal", "intensity", "sigma", "count",
                          "sum_signal", "sum_variance", "sum_normalization",
                          "sum_normalization2", "std", "sem")
        else:
            res = ai.integrate1d_ng(data, npt, **common)
            out_fields = ("radial", "intensity", "sigma", "count", "sum_signal",
                          "sum_variance", "sum_normalization",
                          "sum_normalization2", "std", "sem")

        for field in out_fields:
            v = getattr(res, field, None)
            if isinstance(v, np.ndarray):
                _save(arrays, out_dir, f"out_{field}", v)

        npt_cfg = ({"npt_rad": npt_rad, "npt_azim": npt_azim,
                    "azim_scale": float(units.CHI_DEG.scale),
                    "pos1_period": float(units.CHI_DEG.period),
                    "chi_disc_at_pi": True}
                   if dim == 2 else {"npt": npt})
        manifest = {
            "dataset": key,
            "pyfai_version": pyFAI.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "config": {
                "dim": dim, **npt_cfg, "unit": unit,
                "unit_scale": float(units.to_unit(unit).scale),
                "method": list(method),
                "error_model": error_model,
                "error_model_code": int(em),
                "manage_variance": _manage_variance(method, dim),
                "correct_solid_angle": True,
                "polarization_factor": None,
                "normalization_factor": norm_factor,
                "monitor": monitor,
                "has_dark": True, "has_flat": True, "has_user_mask": True,
                "has_variance": with_var,
                "radial_range": None, "azimuth_range": None,
                "dummy": float(dummy_v),
                "delta_dummy": None if delta_dummy_v is None else float(delta_dummy_v),
            },
            "geometry": geom, "detector": det, "arrays": arrays,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  wrote {key}  ({len(arrays)} arrays, "
              f"manage_variance={manifest['config']['manage_variance']}, "
              f"norm={norm_factor:.6e})")


if __name__ == "__main__":
    main()
