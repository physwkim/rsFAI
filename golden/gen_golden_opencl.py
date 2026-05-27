#!/usr/bin/env python3
"""Generate OpenCL (GPU) golden datasets for the rsFAI Phase-2 backend.

Run in the `daq` conda env (pyFAI + pyopencl, on the Apple M4 Pro GPU):

    PYOPENCL_COMPILER_OUTPUT=0 OMP_NUM_THREADS=1 \
        /Users/stevek/mamba/envs/daq/bin/python golden/gen_golden_opencl.py

Unlike ``gen_golden.py`` (the bit-exact *cython* generator), this drives
pyFAI's **OpenCL** integrators (``OCL_CSR_Integrator``) and dumps what the GPU
actually produced. The Rust ``rsfai-opencl`` backend reuses pyFAI's own
``.cl`` kernels, so on the same device the result is expected to match these
golden values bit-for-bit; the validation gate is relative error <= 1e-6 to
allow for the cross-work-group reduction order (see doc/bit-exact-ladder.md).

Per dataset (``golden/datasets/<name>/``) it emits, as ``.npy`` + JSON:

  * inputs        : image (raw dtype), mask (int8), correction arrays the GPU
                    consumed (solidangle, polarization, ...)
  * CSR matrix    : the exact (data, indices, indptr) the OCL integrator was
                    built from (csr_integr.lut) — the Rust side feeds the same
  * opencl_params : EVERY scalar kernel argument the OCL integrator used
                    (corrections4a + csr_integrate4), captured from the live
                    ``cl_kernel_args`` so the Rust orchestration replicates them
                    without guessing, plus wg_min / image size / bins / empty
  * golden out    : every field of the Integrate1d/2dResult the GPU returned
  * manifest.json : config + provenance (pyFAI/pyopencl versions, device)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "0")

import json
import platform
import shutil
from pathlib import Path

import numpy as np

import pyFAI
import fabio
import pyopencl
from pyFAI.test.utilstest import UtilsTest

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"

# Scalar (non-buffer) corrections4a arguments to capture verbatim. These are
# the int8/float32 flags and constants the GPU preprocessing actually used.
CORR_SCALARS = (
    "dtype", "error_model", "do_dark", "do_dark_variance", "do_flat",
    "do_solidangle", "do_polarization", "do_absorption", "do_mask",
    "do_dummy", "dummy", "delta_dummy", "normalization_factor",
    "apply_normalization",
)
# corrections4 (LUT path) scalars: same as corrections4a minus the image dtype.
# The LUT path pre-casts the image to float (s32_to_float), so corrections4
# reads a float buffer and takes no dtype argument.
CORR4_SCALARS = tuple(k for k in CORR_SCALARS if k != "dtype")
# Scalar csr_integrate4 arguments.
INT_SCALARS = ("nbins", "empty", "error_model")


def _save(arrays_meta, out_dir, name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(out_dir / f"{name}.npy", arr)
    arrays_meta[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}


def _slug(s):
    return (
        str(s).replace("^", "").replace("-", "m").replace("/", "_")
        .replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
    )


def _scalar(v):
    """Convert a numpy scalar kernel argument to a JSON-serialisable Python value."""
    arr = np.asarray(v)
    if arr.dtype.kind == "f":
        return float(arr)
    return int(arr)


def _find_ocl_integrator(ai, class_name):
    """Return the named OCL integrator engine instance from a populated ai."""
    for _method, engine_wrap in ai.engines.items():
        eng = getattr(engine_wrap, "engine", engine_wrap)
        if eng.__class__.__name__ == class_name:
            return eng
    raise RuntimeError(f"no {class_name} engine found after integration")


def generate(detector_name, poni_image, configs):
    poni_name, image_name = poni_image
    poni_path = UtilsTest.getimage(poni_name)
    data = fabio.open(UtilsTest.getimage(image_name)).data
    shape = data.shape

    for cfg in configs:
        # Fresh AzimuthalIntegrator per config: ai.engines accumulates engines
        # across runs, so reusing one ai would let _find_ocl_integrator pick a
        # previous config's integrator. Reloading isolates each config's engine.
        ai = pyFAI.load(poni_path)
        geom = {k: float(getattr(ai, k)) for k in
                ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength")}
        det = {"name": ai.detector.name, "pixel1": float(ai.detector.pixel1),
               "pixel2": float(ai.detector.pixel2), "shape": list(shape),
               "orientation": int(ai.detector.orientation)}
        dim = cfg.get("dim", 1)
        unit = cfg["unit"]
        method = tuple(cfg["method"])
        error_model = cfg.get("error_model")
        correct_solid_angle = cfg.get("correct_solid_angle", True)
        polarization_factor = cfg.get("polarization_factor")
        normalization_factor = cfg.get("normalization_factor", 1.0)
        radial_range = cfg.get("radial_range")
        azimuth_range = cfg.get("azimuth_range")
        if dim == 2:
            npt_rad, npt_azim = cfg["npt_rad"], cfg["npt_azim"]
            npt_slug = f"npt{npt_rad}x{npt_azim}"
        else:
            npt = cfg["npt"]
            npt_slug = f"npt{npt}"

        key = "__".join([_slug(detector_name), "-".join(method), _slug(unit),
                         npt_slug, f"err{error_model or 'none'}"])
        out_dir = DATASETS / key
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        arrays = {}

        # ---- Inputs (exactly what the GPU consumes) ---------------------
        shutil.copyfile(poni_path, out_dir / "geometry.poni")
        _save(arrays, out_dir, "image", data)
        mask = ai.create_mask(data, mask=None).astype(np.int8)
        _save(arrays, out_dir, "mask", mask)
        solidangle = ai.solidAngleArray(shape) if correct_solid_angle else None
        if solidangle is not None:
            _save(arrays, out_dir, "solidangle", solidangle)
        polarization = (ai.polarization(shape, factor=polarization_factor)
                        if polarization_factor is not None else None)
        if polarization is not None:
            _save(arrays, out_dir, "polarization", polarization)

        # ---- Run the OpenCL integration ---------------------------------
        common = dict(unit=unit, method=method,
                      correctSolidAngle=correct_solid_angle,
                      error_model=error_model,
                      polarization_factor=polarization_factor,
                      normalization_factor=normalization_factor,
                      radial_range=radial_range, azimuth_range=azimuth_range)
        if dim == 2:
            res = ai.integrate2d_ng(data, npt_rad, npt_azim, **common)
            out_fields = ("radial", "azimuthal", "intensity", "sigma", "count",
                          "sum_signal", "sum_variance", "sum_normalization",
                          "sum_normalization2", "std", "sem")
        else:
            res = ai.integrate1d_ng(data, npt, **common)
            out_fields = ("radial", "intensity", "sigma", "count",
                          "sum_signal", "sum_variance", "sum_normalization",
                          "sum_normalization2", "std", "sem")

        for field in out_fields:
            v = getattr(res, field, None)
            if isinstance(v, np.ndarray):
                _save(arrays, out_dir, f"out_{field}", v)

        # ---- The OCL integrator's exact state ---------------------------
        # algo is method[1] ("csr" or "lut"); the two integrators differ in the
        # sparse representation, the preproc kernel (corrections4a has a dtype
        # arg, corrections4 does not), and the launch geometry.
        algo = method[1]
        if algo == "csr":
            integr = _find_ocl_integrator(ai, "OCL_CSR_Integrator")
            _save(arrays, out_dir, "csr_data", np.asarray(integr._data))
            _save(arrays, out_dir, "csr_indices", np.asarray(integr._indices))
            _save(arrays, out_dir, "csr_indptr", np.asarray(integr._indptr))
            corr_args = integr.cl_kernel_args["corrections4a"]
            int_args = integr.cl_kernel_args["csr_integrate4"]
            wg_min, wg_max = integr.workgroup_size["csr_integrate4"]
            algo_params = {
                "corrections4a": {k: _scalar(corr_args[k]) for k in CORR_SCALARS},
                "csr_integrate4": {k: _scalar(int_args[k]) for k in INT_SCALARS},
                # csr_integrate4 launch (azim_csr.integrate_ng): wg = wg_min,
                # global = bins * wg_min, local = wg_min, shared = 32 * wg_min.
                "wg_min": int(wg_min),
                "wg_max": int(wg_max),
                # The all-ones LUT short-circuit (azim_csr __init__): when every
                # coef is 1.0, pyFAI passes a NULL coefs buffer (kernel uses 1.0).
                "data_is_ones": integr.cl_mem.get("data") is None,
            }
        elif algo == "lut":
            integr = _find_ocl_integrator(ai, "OCL_LUT_Integrator")
            # Densified LUT (bins, lut_size) of struct {idx:int32, coef:float32}.
            # Dump the two fields separately; the Rust side transposes them for
            # the GPU (lut.T) and re-interleaves the struct.
            _save(arrays, out_dir, "lut_idx", np.ascontiguousarray(integr._lut["idx"]))
            _save(arrays, out_dir, "lut_coef", np.ascontiguousarray(integr._lut["coef"]))
            corr_args = integr.cl_kernel_args["corrections4"]
            algo_params = {
                "corrections4": {k: _scalar(corr_args[k]) for k in CORR4_SCALARS},
                # lut_integrate4 launch (azim_lut.integrate_ng): one thread per
                # bin, global = ceil(bins/block_size)*block_size, local=block_size.
                "block_size": int(integr.BLOCK_SIZE),
                "lut_size": int(integr.lut_size),
            }
        elif algo == "histogram":
            klass = "OCL_Histogram2d" if dim == 2 else "OCL_Histogram1d"
            integr = _find_ocl_integrator(ai, klass)
            # No sparse matrix: the histogram consumes the per-pixel radial (and
            # azimuthal) position arrays directly and atomic-adds into the bins.
            # Dump the exact float32 buffers the GPU received (send_buffer casts
            # to float32); the radial_mini/maxi/azim scalars are captured live so
            # the bin assignment matches without recomputation.
            _save(arrays, out_dir, "radial",
                  np.ascontiguousarray(integr.radial, dtype=np.float32))
            _save(arrays, out_dir, "azimuthal",
                  np.ascontiguousarray(integr.azimuthal, dtype=np.float32))
            corr_args = integr.cl_kernel_args["corrections4"]
            # The per-CALL histogram-preproc args (integrate() mutates this dict
            # in place): radial_mini/maxi and azim_mini/maxi reflect the actual
            # radial_range/azimuth_range used. check_azim is the per-call flag
            # (0 here: integrate1d passes no azimuth_range — so it is NOT the
            # instance attribute, which is 1 merely because an azimuthal array
            # exists). The 2D preproc kernel has no check_azim (it always
            # range-checks azimuth).
            hk = "histogram_2d_preproc" if dim == 2 else "histogram_1d_preproc"
            hist_args = integr.cl_kernel_args[hk]
            algo_params = {
                "corrections4": {k: _scalar(corr_args[k]) for k in CORR4_SCALARS},
                # histogram_{1d,2d}_preproc launch: global = ceil(size/block)*block
                # (one thread per pixel); memset_histograms / histogram_postproc:
                # global = ceil(bins/block)*block (one thread per bin).
                "block_size": int(integr.BLOCK_SIZE),
                "radial_mini": _scalar(hist_args["radial_mini"]),
                "radial_maxi": _scalar(hist_args["radial_maxi"]),
                "azim_mini": _scalar(hist_args["azim_mini"]),
                "azim_maxi": _scalar(hist_args["azim_maxi"]),
                **({"check_azim": int(_scalar(hist_args["check_azim"]))}
                   if dim == 1 else {}),
            }
        else:
            raise RuntimeError(f"unsupported algo {algo!r}")

        opencl_params = {
            "algo": algo,
            "bins": int(integr.bins),
            "image_size": int(integr.size),
            "empty": float(integr.empty),
            # Dimensionality + 2D cell layout. integrate_ng packs 2D as a flat
            # (bins_rad * bins_azim) sparse matrix (cell = rad*bins_azim + azim,
            # radial-major) then reshapes (bins_rad, bins_azim).T → (azim, rad).
            "dim": dim,
            **({"bins_rad": npt_rad, "bins_azim": npt_azim} if dim == 2 else {}),
            **algo_params,
        }
        with open(out_dir / "opencl_params.json", "w") as f:
            json.dump(opencl_params, f, indent=2)

        # ---- Manifest ---------------------------------------------------
        manifest = {
            "dataset": key,
            "detector_name": detector_name,
            "backend": "opencl",
            "pyfai_version": pyFAI.version,
            "numpy_version": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "pyopencl_version": pyopencl.VERSION_TEXT,
            "opencl_device": integr.ctx.devices[0].name.strip(),
            "platform": platform.platform(),
            "tolerance_note": (
                "OpenCL golden: GPU doubleword (two-f32 Kahan) accumulation. The "
                "Rust backend reuses pyFAI's own .cl kernels on the same device, "
                "validated at relative error <= 1e-6 (cross-work-group reduction "
                "order). See doc/bit-exact-ladder.md."
            ),
            "config": {
                "dim": dim, "unit": unit, "method": list(method),
                "error_model": error_model,
                "correct_solid_angle": correct_solid_angle,
                "polarization_factor": polarization_factor,
                "normalization_factor": normalization_factor,
                "radial_range": radial_range, "azimuth_range": azimuth_range,
                **({"npt_rad": npt_rad, "npt_azim": npt_azim} if dim == 2
                   else {"npt": npt}),
            },
            "geometry": geom,
            "detector": det,
            "arrays": arrays,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  wrote {key}  ({len(arrays)} arrays, algo={algo}, "
              f"bins={opencl_params['bins']}, dim={dim})")


def main():
    DATASETS.mkdir(parents=True, exist_ok=True)
    print(f"pyFAI {pyFAI.version}, pyopencl {pyopencl.VERSION_TEXT}, "
          f"numpy {np.__version__}")
    generate(
        "Pilatus1M",
        ("Pilatus1M.poni", "Pilatus1M.edf"),
        configs=[
            {
                # First OpenCL tuple: no-split CSR, 1D, Poisson. The LUT coefs are
                # all 1.0 (NULL-coefs kernel path); on the M4 Pro GPU the wg_max
                # for csr_integrate4 is 256 (!=1), so pyFAI runs the tree-reduction
                # csr_integrate4 (NOT csr_integrate4_single) with wg = wg_min = 32.
                "npt": 1000,
                "unit": "q_nm^-1",
                "method": ("no", "csr", "opencl"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": None,
            },
            # ---- Remaining CSR tuples (all OCL_CSR_Integrator) --------------
            # bbox/full carry fractional coefs (data_is_ones=False, real coefs
            # buffer) and polarization (do_polarization=1). 2D adds the
            # (bins_rad, bins_azim).T → (azim, rad) output packaging; the kernel
            # call is identical (the 2D LUT flattens cells to bin0*bins1+bin1).
            {
                "npt": 1000, "unit": "q_nm^-1",
                "method": ("bbox", "csr", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "npt": 1000, "unit": "q_nm^-1",
                "method": ("full", "csr", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("no", "csr", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("bbox", "csr", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("full", "csr", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            # ---- LUT tuples (all OCL_LUT_Integrator) -----------------------
            # Same NG pipeline shape as CSR but a densified+transposed LUT and a
            # one-thread-per-bin integrate kernel; corrections4 has no dtype arg
            # (the image is pre-cast to float). 2D packaging is identical to CSR.
            {
                "npt": 1000, "unit": "q_nm^-1",
                "method": ("no", "lut", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "npt": 1000, "unit": "q_nm^-1",
                "method": ("bbox", "lut", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "npt": 1000, "unit": "q_nm^-1",
                "method": ("full", "lut", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("no", "lut", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("bbox", "lut", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("full", "lut", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            # ---- Histogram tuples (OCL_Histogram1d / OCL_Histogram2d) -------
            # Atomic-add scatter (no sparse matrix): per-pixel radial/azimuthal
            # arrays are histogrammed directly. The atomic_cmpxchg order is
            # non-deterministic, so this path is tolerance-only (rel <= 1e-6);
            # the doubleword Kahan accumulation bounds the error to ~f64.
            {
                "npt": 1000, "unit": "q_nm^-1",
                "method": ("no", "histogram", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                "dim": 2, "npt_rad": 100, "npt_azim": 36, "unit": "q_nm^-1",
                "method": ("no", "histogram", "opencl"),
                "error_model": "poisson", "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
        ],
    )
    print("done.")


if __name__ == "__main__":
    main()
