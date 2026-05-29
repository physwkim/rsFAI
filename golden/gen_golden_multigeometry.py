#!/usr/bin/env python3
"""Golden generator for ``MultiGeometry`` (``multi_geometry.py``) — several
detector geometries integrated into one shared radial/azimuth grid and combined
by the sequential weighted ``union`` + ``__recalculate_means__``
(``containers.py``).

This gates the ``rsfai-multigeometry`` crate end-to-end:

  * per-geometry monitor scaling ``monitor * pixel1*pixel2/dist**2`` (f64), with
    ``correctSolidAngle`` on **and** off;
  * the common range *guessed* as ``(min, max)`` over each geometry's scaled
    ``array_from_unit`` (radial) / ``CHI_DEG`` (azimuth) — recorded so the Rust
    verifier can cross-check its own guess bit-exactly before integrating;
  * the left-fold ``union`` over ≥2 geometries, including the AZIMUTHAL crossed
    term, the variance/norm² combine (poisson), and the plain combine (no model);
  * ``__recalculate_means__`` (intensity always; sem/std/sigma when variance).

Inputs (3 Pilatus1M geometries, 3 frames, a shared user mask + flat, per-geometry
monitors) are written **once** to ``datasets_multigeometry/inputs/`` and shared by
every config; only each config's ``out_*.npy`` + ``manifest.json`` differ. Method
is the MG default ``("full","histogram","cython")`` (serial split-histogram ⇒
bit-exact) plus a ``("bbox","csr","cython")`` variant — neither is the no-split
histogram, so every field is gated bit-exact (0-ULP).

Run in the ``daq`` conda env, single-thread (the bit-exact gate):

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/gen_golden_multigeometry.py
"""

import itertools
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyFAI  # noqa: E402
from pyFAI import units  # noqa: E402
from pyFAI.containers import ErrorModel  # noqa: E402
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402
from pyFAI.multi_geometry import MultiGeometry  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "datasets_multigeometry"
INPUTS = OUT_ROOT / "inputs"

DETECTOR = "Pilatus1M"
SHAPE = (1043, 981)

# Three distinct geometries (dist / poni / rot offsets) so each frame lands at a
# different beam-centre — the guessed common range then spans the union of all
# three, exercising the cross-geometry range guess and the weighted combine.
GEOMETRIES = [
    dict(dist=1.58323111834, poni1=0.0334170169115, poni2=0.0412277798782,
         rot1=0.00648735642526, rot2=0.00755810191106, rot3=4.12987220385e-08),
    dict(dist=1.58323111834 * 1.05, poni1=0.0534170169115, poni2=0.0312277798782,
         rot1=0.01648735642526, rot2=0.00755810191106, rot3=4.12987220385e-08),
    dict(dist=1.58323111834 * 0.95, poni1=0.0234170169115, poni2=0.0612277798782,
         rot1=0.00648735642526, rot2=0.01755810191106, rot3=4.12987220385e-08),
]
WAVELENGTH = 1.0e-10
# Per-geometry normalization monitors (distinct ⇒ each geometry's denominator
# differs, so a wrong per-geometry carry shows up in the combine).
MONITORS = [1.0, 2.7, 0.6]

# Matrix axes. Method ∈ {MG default full-histogram, a CSR variant}; both serial
# ⇒ bit-exact. r_mm/2th_deg/q exercise the per-unit radial range guess.
UNITS = ["q_nm^-1", "2th_deg", "r_mm"]
ERROR_MODELS = [None, "poisson", "azimuthal"]
CORRECT_SOLID_ANGLE = [True, False]
METHODS = [("full", "histogram", "cython"), ("bbox", "csr", "cython")]
NPT_1D = 500
NPT_RAD_2D, NPT_AZIM_2D = 100, 60

# rsFAI ErrorModel codes (dtype.rs): No=0, Variance=1, Poisson=2, Azimuthal=3.
EM_CODE = {None: 0, "poisson": 2, "azimuthal": 3}


def _save(meta, out_dir, name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(out_dir / f"{name}.npy", arr)
    meta[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}


def build_ais():
    det = pyFAI.detector_factory(DETECTOR)
    ais = []
    for g in GEOMETRIES:
        ai = AzimuthalIntegrator(detector=det, wavelength=WAVELENGTH, **g)
        ais.append(ai)
    return ais


def make_inputs():
    """Write the shared inputs once; return (images, user_mask, flat)."""
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    INPUTS.mkdir(parents=True)

    ais = build_ais()
    images = []
    for i, ai in enumerate(ais):
        rng = np.random.default_rng(20260529 + i)
        # Positive f32 counts (smooth radial bump + noise) so poisson variance
        # `max(data,1)` is meaningful and the flat/monitor actually scale it.
        img = (rng.random(SHAPE, dtype=np.float64) * 800.0 + 25.0).astype(np.float32)
        images.append(img)
        ai.save(str(INPUTS / f"geometry_{i}.poni"))
        np.save(INPUTS / f"image_{i}.npy", img)

    # Shared user mask (REPLACES the detector mask) + shared flat, broadcast by
    # pyFAI to every geometry (lst_mask/lst_flat as a single ndarray).
    user_mask = np.zeros(SHAPE, dtype=np.int8)
    user_mask[200:260, 300:380] = 1
    np.save(INPUTS / "user_mask.npy", user_mask)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    flat = (1.0 + 0.05 * (xx / SHAPE[1]) + 0.03 * (yy / SHAPE[0])).astype(np.float32)
    np.save(INPUTS / "flat.npy", flat)

    det = ais[0].detector
    inputs_meta = {
        "detector": DETECTOR,
        "shape": list(SHAPE),
        "n_geometry": len(ais),
        "wavelength": WAVELENGTH,
        "monitors": MONITORS,
        "ponis": [f"geometry_{i}.poni" for i in range(len(ais))],
        "images": [f"image_{i}.npy" for i in range(len(ais))],
        "pixel1": float(det.pixel1),
        "pixel2": float(det.pixel2),
        "user_mask": "user_mask.npy",
        "flat": "flat.npy",
    }
    with open(INPUTS / "inputs.json", "w") as f:
        json.dump(inputs_meta, f, indent=2)
    return ais, images, user_mask, flat


def main():
    print(f"pyFAI {pyFAI.version}, numpy {np.__version__}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    _, images, user_mask, flat = make_inputs()

    written, skipped = 0, 0
    for dim, unit, em, csa, method in itertools.product(
            (1, 2), UNITS, ERROR_MODELS, CORRECT_SOLID_ANGLE, METHODS):
        ukey = unit.replace("^", "").replace("-", "m").replace("/", "_")
        emkey = "none" if em is None else em
        mkey = "-".join(method)
        key = (f"MultiGeometry__{mkey}__{ukey}__dim{dim}"
               f"__err{emkey}__csa{int(csa)}")
        out_dir = OUT_ROOT / key

        # Fresh MultiGeometry per config: it caches the guessed range on `self`,
        # so a shared instance would leak the first config's range into the rest.
        # chi_disc=180 (default) ⇒ disc at π, matching rsFAI's chiDiscAtPi=true.
        ais = build_ais()
        mg = MultiGeometry(ais, unit=unit, radial_range=None, azimuth_range=None,
                           empty=0.0, chi_disc=180, threadpoolsize=0)
        try:
            common = dict(correctSolidAngle=csa, error_model=em,
                          polarization_factor=None,
                          normalization_factor=list(MONITORS),
                          lst_mask=user_mask, lst_flat=flat, method=method)
            if dim == 1:
                res = mg.integrate1d(images, npt=NPT_1D, **common)
                out_fields = ("radial", "intensity", "sigma", "count",
                              "sum_signal", "sum_variance", "sum_normalization",
                              "sum_normalization2", "std", "sem")
            else:
                res = mg.integrate2d(images, npt_rad=NPT_RAD_2D, npt_azim=NPT_AZIM_2D,
                                     **common)
                out_fields = ("radial", "azimuthal", "intensity", "sigma", "count",
                              "sum_signal", "sum_variance", "sum_normalization",
                              "sum_normalization2", "std", "sem")
        except Exception as exc:  # noqa: BLE001 — record & skip unsupported combos
            print(f"  SKIP {key}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        out_dir.mkdir(parents=True)
        arrays = {}
        for field in out_fields:
            v = getattr(res, field, None)
            if isinstance(v, np.ndarray):
                _save(arrays, out_dir, f"out_{field}", v)

        # The range the MG actually guessed/used (mutated onto `mg` by integrate).
        rr = mg.radial_range
        ar = mg.azimuth_range
        npt_cfg = ({"npt_rad": NPT_RAD_2D, "npt_azim": NPT_AZIM_2D,
                    "azim_scale": float(units.CHI_DEG.scale),
                    "pos1_period": float(units.CHI_DEG.period),
                    "chi_disc_at_pi": True}
                   if dim == 2 else {"npt": NPT_1D})
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
                "error_model": emkey,
                "error_model_code": EM_CODE[em],
                "correct_solid_angle": csa,
                "polarization_factor": None,
                "n_geometry": len(ais),
                "monitors": list(MONITORS),
                "has_user_mask": True, "has_flat": True,
                "empty": 0.0,
                "radial_range_guessed": [float(rr[0]), float(rr[1])],
                "azimuth_range_guessed": [float(ar[0]), float(ar[1])],
            },
            "arrays": arrays,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        written += 1
        print(f"  wrote {key}  ({len(arrays)} arrays, "
              f"radial_range={rr[0]:.5f}..{rr[1]:.5f})")

    print(f"\nDONE: {written} datasets written, {skipped} skipped.")


if __name__ == "__main__":
    main()
