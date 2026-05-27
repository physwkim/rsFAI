#!/usr/bin/env python3
"""Generate golden datasets for validating the rsFAI Rust port against pyFAI.

Run in the `daq` conda env (which has pyFAI installed):

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/gen_golden.py

For each (detector image, integration config) pair this script emits a
self-contained directory under ``golden/datasets/`` containing, as ``.npy``
files (lossless bit preservation) plus a ``manifest.json``:

  * inputs      : image, mask, geometry (poni params), correction arrays
  * Tier-B      : geometry position arrays (center/delta/corner, chi)
  * Tier-A      : per-pixel preproc output, and (for CSR methods) the sparse
                  matrix (data/indices/indptr)
  * golden out  : every field exposed by the Integrate1dResult

The bit-exact ladder and what each tier must satisfy are documented in
``doc/bit-exact-ladder.md``. Generation is pinned to ``OMP_NUM_THREADS=1`` and
pyFAI's serial Cython path so the accumulation order is deterministic.

NOTE (M0): the golden ``sum_*`` fields are exposed by pyFAI as float32 (it
downcasts the float64 accumulators for storage). Validating the full-precision
float64 accumulators is an M4 refinement; the dtype of each array is recorded in
the manifest so the Rust comparator matches pyFAI's exposed width.
"""

import os

# Must be set before importing pyFAI so the OpenMP kernels honor it.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import platform
import shutil
from pathlib import Path

import numpy as np

import pyFAI
import fabio
from pyFAI.test.utilstest import UtilsTest
from pyFAI.containers import ErrorModel
import pyFAI.ext.preproc as ext_preproc

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"


def _error_model(name):
    """Map an integrate1d error_model string (or None) to pyFAI's ErrorModel."""
    if name is None or str(name).lower() == "no":
        return ErrorModel.NO
    return ErrorModel.parse(name)


def _save(arrays_meta, out_dir, name, arr):
    """Save ``arr`` as ``name.npy`` (C-contiguous) and record its metadata."""
    arr = np.ascontiguousarray(arr)
    np.save(out_dir / f"{name}.npy", arr)
    arrays_meta[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}


def _slug(s):
    return (
        str(s)
        .replace("^", "")
        .replace("-", "m")
        .replace("/", "_")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def generate(detector_name, poni_image, configs):
    """Generate all configs for one detector image.

    :param poni_image: (poni_resource_name, image_resource_name) for UtilsTest.
    """
    poni_name, image_name = poni_image
    poni_path = UtilsTest.getimage(poni_name)
    image_path = UtilsTest.getimage(image_name)

    ai = pyFAI.load(poni_path)
    data = fabio.open(image_path).data
    shape = data.shape

    geom = {
        "dist": float(ai.dist),
        "poni1": float(ai.poni1),
        "poni2": float(ai.poni2),
        "rot1": float(ai.rot1),
        "rot2": float(ai.rot2),
        "rot3": float(ai.rot3),
        "wavelength": float(ai.wavelength),
    }
    det = {
        "name": ai.detector.name,
        "pixel1": float(ai.detector.pixel1),
        "pixel2": float(ai.detector.pixel2),
        "shape": list(shape),
    }

    for cfg in configs:
        npt = cfg["npt"]
        unit = cfg["unit"]
        method = tuple(cfg["method"])
        error_model = cfg.get("error_model")
        correct_solid_angle = cfg.get("correct_solid_angle", True)
        polarization_factor = cfg.get("polarization_factor")
        normalization_factor = cfg.get("normalization_factor", 1.0)
        radial_range = cfg.get("radial_range")
        azimuth_range = cfg.get("azimuth_range")

        key = "__".join(
            [
                _slug(detector_name),
                "-".join(method),
                _slug(unit),
                f"npt{npt}",
                f"err{error_model or 'none'}",
            ]
        )
        out_dir = DATASETS / key
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        arrays = {}

        # ---- Inputs -----------------------------------------------------
        _save(arrays, out_dir, "image", data)
        mask = ai.create_mask(data, mask=None).astype(np.int8)  # 1 = masked
        _save(arrays, out_dir, "mask", mask)
        shutil.copyfile(poni_path, out_dir / "geometry.poni")

        solidangle = ai.solidAngleArray(shape) if correct_solid_angle else None
        if solidangle is not None:
            _save(arrays, out_dir, "solidangle", solidangle)
        polarization = (
            ai.polarization(shape, factor=polarization_factor)
            if polarization_factor is not None
            else None
        )
        if polarization is not None:
            _save(arrays, out_dir, "polarization", polarization)

        # ---- Tier-B geometry position arrays ----------------------------
        _save(arrays, out_dir, "pos0_center", ai.center_array(shape, unit=unit))
        _save(arrays, out_dir, "pos0_delta", ai.delta_array(shape, unit=unit))
        _save(arrays, out_dir, "chi_center", ai.center_array(shape, unit="chi_rad"))
        _save(arrays, out_dir, "chi_delta", ai.delta_array(shape, unit="chi_rad"))
        _save(arrays, out_dir, "corners", ai.corner_array(shape, unit=unit, scale=False))

        # ---- Tier-A per-pixel preproc -----------------------------------
        # Reproduce the per-pixel (signal, variance, norm, count) the engine
        # consumes. dtype defaults to float32 (data_t) — matching pyFAI.
        em = _error_model(error_model)
        em_code = int(em)
        preq = ext_preproc.preproc(
            data.astype(np.float32),
            solidangle=solidangle,
            polarization=polarization,
            normalization_factor=normalization_factor,
            mask=mask,
            error_model=em,
            split_result=4,  # -> (signal, variance, norm, count)
        )
        _save(arrays, out_dir, "preproc", preq)

        # ---- Run the integration ----------------------------------------
        res = ai.integrate1d_ng(
            data,
            npt,
            unit=unit,
            method=method,
            correctSolidAngle=correct_solid_angle,
            error_model=error_model,
            polarization_factor=polarization_factor,
            normalization_factor=normalization_factor,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
        )

        # ---- Golden output (every exposed field) ------------------------
        for field in (
            "radial",
            "intensity",
            "sigma",
            "count",
            "sum_signal",
            "sum_variance",
            "sum_normalization",
            "sum_normalization2",
            "std",
            "sem",
        ):
            v = getattr(res, field, None)
            if isinstance(v, np.ndarray):
                _save(arrays, out_dir, f"out_{field}", v)

        # ---- Tier-A sparse matrix (CSR methods only) --------------------
        # Only CSR configs build a CSR engine; histogram/"no" configs add none,
        # so this loop simply finds nothing for them.
        for m, engine_wrap in ai.engines.items():
            if "CSR" not in str(m):
                continue
            eng = getattr(engine_wrap, "engine", engine_wrap)
            if all(hasattr(eng, a) for a in ("data", "indices", "indptr")):
                _save(arrays, out_dir, "csr_data", np.asarray(eng.data))
                _save(arrays, out_dir, "csr_indices", np.asarray(eng.indices))
                _save(arrays, out_dir, "csr_indptr", np.asarray(eng.indptr))
                break

        # ---- Manifest ---------------------------------------------------
        manifest = {
            "dataset": key,
            "detector_name": detector_name,
            "pyfai_version": pyFAI.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "provenance_note": (
                "pyFAI installed from ESRF prebuilt cp314 macOS-arm64 wheel "
                "(not a local source build); see doc/bit-exact-ladder.md"
            ),
            "config": {
                "npt": npt,
                "unit": unit,
                "method": list(method),
                "error_model": error_model,
                "error_model_code": em_code,
                "correct_solid_angle": correct_solid_angle,
                "polarization_factor": polarization_factor,
                "normalization_factor": normalization_factor,
                "radial_range": radial_range,
                "azimuth_range": azimuth_range,
            },
            "geometry": geom,
            "detector": det,
            # Aspirational budget; M1 measures and fills the real ULP figures.
            "ulp_budget": {"pos0_center": 0, "chi_center": 0},
            "arrays": arrays,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  wrote {key}  ({len(arrays)} arrays)")


def main():
    DATASETS.mkdir(parents=True, exist_ok=True)
    print(f"pyFAI {pyFAI.version}, numpy {np.__version__}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    generate(
        "Pilatus1M",
        ("Pilatus1M.poni", "Pilatus1M.edf"),
        configs=[
            {
                "npt": 1000,
                "unit": "q_nm^-1",
                "method": ("no", "histogram", "cython"),
                "error_model": None,
                "correct_solid_angle": True,
                "polarization_factor": None,
            },
            {
                "npt": 1000,
                "unit": "2th_deg",
                "method": ("bbox", "csr", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
        ],
    )
    print("done.")


if __name__ == "__main__":
    main()
