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
from pyFAI import units
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
        "orientation": int(ai.detector.orientation),
    }

    for cfg in configs:
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
            npt_rad = cfg["npt_rad"]
            npt_azim = cfg["npt_azim"]
            npt_slug = f"npt{npt_rad}x{npt_azim}"
        else:
            npt = cfg["npt"]
            npt_slug = f"npt{npt}"

        key = "__".join(
            [
                _slug(detector_name),
                "-".join(method),
                _slug(unit),
                npt_slug,
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
        # The radial array the integration engines actually bin/build on is the
        # UNSCALED centre (`center_array(scale=False)`); the reported position is
        # `engine_position * unit.scale`. For q_nm^-1 the scale is 1.0 so the two
        # coincide, but for 2th_deg (scale ≈ 57.3) they differ — and the unscaled
        # value cannot be recovered from the scaled one by division without
        # losing bits. Dump it explicitly as the Tier-A engine input.
        _save(arrays, out_dir, "pos0_center_unscaled",
              ai.center_array(shape, unit=unit, scale=False))
        _save(arrays, out_dir, "pos0_delta", ai.delta_array(shape, unit=unit))
        _save(arrays, out_dir, "chi_center", ai.center_array(shape, unit="chi_rad"))
        _save(arrays, out_dir, "chi_delta", ai.delta_array(shape, unit="chi_rad"))
        _save(arrays, out_dir, "corners", ai.corner_array(shape, unit=unit, scale=False))

        # calc_pos_zyx checkpoint: lab coords (z,y,x) and the detector pixel
        # centers (p1,p2[,p3]) that feed the rotation transform. Lets the Rust
        # port validate calc_pos_zyx in isolation (Tier A, given pixel centers),
        # before the detector model (M2) reproduces the centers.
        _save(arrays, out_dir, "pos_zyx", ai.position_array(shape, corners=False))
        # Dump the *exact* float64 pixel centers position_array feeds the
        # transform: it builds float64 index grids via numpy.fromfunction, so we
        # must too (the no-arg call returns float32 centers with different bits).
        d1 = np.fromfunction(lambda i, j: i, tuple(shape), dtype=np.float64)
        d2 = np.fromfunction(lambda i, j: j, tuple(shape), dtype=np.float64)
        p1c, p2c, p3c = ai.detector.calc_cartesian_positions(d1, d2)
        _save(arrays, out_dir, "pixel_p1", p1c)
        _save(arrays, out_dir, "pixel_p2", p2c)
        if p3c is not None:
            _save(arrays, out_dir, "pixel_p3", p3c)

        # ---- Tier-A per-pixel preproc -----------------------------------
        # Reproduce the per-pixel (signal, variance, norm, count) the engine
        # consumes. dtype defaults to float32 (data_t) — matching pyFAI.
        em = _error_model(error_model)
        em_code = int(em)
        # The integrator masks pixels at the detector's dummy value (Pilatus
        # marks dead/gap pixels as -2 with a ±1.5 tolerance). integrate1d_ng
        # derives these via _normalize_dummies(None, None, data) when the caller
        # passes no dummy, then feeds them to preproc. Reproduce that *exactly*
        # so the dumped preproc is the engine's true input — omitting the dummy
        # leaves a handful of dead pixels valid and shifts the binned sums.
        dummy_v, delta_dummy_v = ai._normalize_dummies(None, None, data)
        preq = ext_preproc.preproc(
            data.astype(np.float32),
            solidangle=solidangle,
            polarization=polarization,
            normalization_factor=normalization_factor,
            mask=mask,
            dummy=dummy_v,
            delta_dummy=delta_dummy_v,
            error_model=em,
            split_result=4,  # -> (signal, variance, norm, count)
        )
        _save(arrays, out_dir, "preproc", preq)

        # ---- Run the integration ----------------------------------------
        if dim == 2:
            res = ai.integrate2d_ng(
                data,
                npt_rad,
                npt_azim,
                unit=unit,
                method=method,
                correctSolidAngle=correct_solid_angle,
                error_model=error_model,
                polarization_factor=polarization_factor,
                normalization_factor=normalization_factor,
                radial_range=radial_range,
                azimuth_range=azimuth_range,
            )
            # 2D fields are (npt_azim, npt_rad). out_radial/out_azimuthal are the
            # 1D scaled bin centres (radial * pos0_scale, azimuthal * pos1_scale).
            out_fields = (
                "radial",
                "azimuthal",
                "intensity",
                "sigma",
                "count",
                "sum_signal",
                "sum_variance",
                "sum_normalization",
                "sum_normalization2",
                "std",
                "sem",
            )
        else:
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
            out_fields = (
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
            )

        # ---- Golden output (every exposed field) ------------------------
        for field in out_fields:
            v = getattr(res, field, None)
            if isinstance(v, np.ndarray):
                _save(arrays, out_dir, f"out_{field}", v)

        # ---- Tier-A sparse matrix (CSR methods only) --------------------
        # Match the engine to THIS config's (split, algo, dim). ai.engines
        # accumulates engines across every config run on this `ai`, so a bare
        # "first CSR engine" pick would dump the bbox matrix for a full-split
        # dataset (or vice-versa) once both have run. The method key carries
        # split_lower / algo_lower / dimension; key off all three (a 1D and a 2D
        # bbox-CSR engine both have split_lower=="bbox").
        if method[1] == "csr":
            for m, engine_wrap in ai.engines.items():
                if (getattr(m, "split_lower", None) != method[0]
                        or getattr(m, "algo_lower", None) != "csr"
                        or getattr(m, "dimension", None) != dim):
                    continue
                eng = getattr(engine_wrap, "engine", engine_wrap)
                if all(hasattr(eng, a) for a in ("data", "indices", "indptr")):
                    _save(arrays, out_dir, "csr_data", np.asarray(eng.data))
                    _save(arrays, out_dir, "csr_indices", np.asarray(eng.indices))
                    _save(arrays, out_dir, "csr_indptr", np.asarray(eng.indptr))
                    break

        # ---- Manifest ---------------------------------------------------
        if dim == 2:
            npt_cfg = {
                "npt_rad": npt_rad,
                "npt_azim": npt_azim,
                # The azimuthal engine input is chi in radians (chi_center.npy);
                # the reported azimuthal is bin_centers1 * azim_scale (CHI_DEG).
                "azim_scale": float(units.CHI_DEG.scale),
                # pos1_period > 0 turns on the [-π, π] azimuthal clip (chiDiscAtPi
                # default True); histogram2d does not otherwise use the period.
                "pos1_period": float(units.CHI_DEG.period),
                "chi_disc_at_pi": True,
            }
        else:
            npt_cfg = {"npt": npt}
        manifest = {
            "dataset": key,
            "detector_name": detector_name,
            "pyfai_version": pyFAI.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "build": {
                # pyFAI rebuilt from the local ~/codes/pyFAI source into daq with
                # FMA contraction disabled, so the Cython geometry evaluates the
                # bare IEEE-754 expression (a*b + c*d - e*f) with no fused
                # multiply-add. That makes the algebraic transform (calc_pos_zyx)
                # bitwise-reproducible by Rust's plain f64 ops. See
                # doc/bit-exact-ladder.md.
                "from_source": True,
                "source_tree": "~/codes/pyFAI",
                "cflags": "-ffp-contract=off",
                "cxxflags": "-ffp-contract=off",
            },
            "provenance_note": (
                "pyFAI rebuilt from local source with -ffp-contract=off (no FMA); "
                "see doc/bit-exact-ladder.md"
            ),
            "config": {
                "dim": dim,
                **npt_cfg,
                "unit": unit,
                # The engine bins/builds on the unscaled radial; the reported
                # position is engine_position * unit_scale (see
                # pos0_center_unscaled). Recorded so the Rust Tier-A tests can
                # apply the single f64 multiply pyFAI does.
                "unit_scale": float(units.to_unit(unit).scale),
                "method": list(method),
                "error_model": error_model,
                "error_model_code": em_code,
                "correct_solid_angle": correct_solid_angle,
                "polarization_factor": polarization_factor,
                "normalization_factor": normalization_factor,
                "radial_range": radial_range,
                "azimuth_range": azimuth_range,
                # Dummy (dead/gap-pixel) masking the integrator applies, derived
                # from the detector. Recorded as f32-exact floats so the Rust
                # preproc test can reproduce the engine's masking (f32→f64→f32
                # round-trips losslessly). delta_dummy may be null (exact match).
                "dummy": float(dummy_v),
                "delta_dummy": None if delta_dummy_v is None else float(delta_dummy_v),
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
                # Poisson histogram: exercises the engine's error-model path —
                # norm² via f64 multiply, variance = max(data, 1), and
                # std/sem via libc double sqrt — which the errnone config
                # cannot (pyFAI exposes no variance/std/sem when do_variance is
                # false). Dumps the full Integrate1dtpl field set.
                "npt": 1000,
                "unit": "q_nm^-1",
                "method": ("no", "histogram", "cython"),
                "error_model": "poisson",
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
            {
                # Full pixel splitting: each pixel's 4 corners are clipped against
                # the radial bins (_recenter handles the chi discontinuity,
                # _integrate1d the trapezoidal overlap). Builds a CSR matrix from
                # the corner array; the apply is the same CsrIntegrator.integrate_ng
                # as the bbox path. Exercises the corner-array build the bbox split
                # cannot.
                "npt": 1000,
                "unit": "2th_deg",
                "method": ("full", "csr", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # No-split CSR: pyFAI's ("no","csr","cython") uses the SAME
                # HistoBBox1d class as bbox-CSR but with delta=None
                # (do_split=False), so each pixel's bounding box collapses to its
                # centre -> one coef-1.0 entry per pixel. Mathematically equals the
                # no-split histogram, but the CSR apply reduces per-bin (ascending
                # pixel index) instead of in pixel-scan order, so the bits differ
                # -> its own golden. Poisson exercises the variance fields.
                "npt": 1000,
                "unit": "q_nm^-1",
                "method": ("no", "csr", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": None,
            },
            {
                # 2D histogram (no split): bins each pixel centre into a
                # (radial, azimuthal) cell via histogram2d_engine. errnone exercises
                # the cnt>0 reduction without the variance branch.
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("no", "histogram", "cython"),
                "error_model": None,
                "correct_solid_angle": True,
                "polarization_factor": None,
            },
            {
                # 2D histogram, Poisson: adds the variance branch (sum_variance,
                # norm_sq via the f32 norm*norm of update_2d_accumulator, std/sem
                # via libc double sqrt).
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("no", "histogram", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # 2D bbox -> CSR, Poisson: builds the 2D LUT (calc_lut_2d) by
                # clipping each pixel's bbox (centre ± delta in both radial and
                # azimuthal) against the (radial, azimuthal) grid, output bin
                # bin0*bins1 + bin1. The apply is the same CsrIntegrator.integrate_ng
                # as 1D, reshaped to (bins0, bins1) and transposed to (azim, rad).
                # chiDiscAtPi defaults True (not forwarded by common.py for
                # HistoBBox2d); pos1_period = CHI_DEG.period acts as the clip flag.
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("bbox", "csr", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # 2D full pixel splitting -> CSR, Poisson: builds the 2D LUT
                # (splitpixel_common.calc_lut_2d) by recentering each pixel's 4
                # corners (_recenter, chi discontinuity), clipping into a small box
                # and sweeping the 4 edges with _integrate2d (whose _calc_area
                # fused type resolves per call site — f64 for same-unit/subsection
                # segments, f32 for segments bounded by the float P local). The
                # apply is the same CsrIntegrator.integrate_ng as 2D bbox. Unlike
                # bbox-2D, common.py forwards chiDiscAtPi (True) and pos1_period =
                # unit1.period (360, applied to radian azimuths — a pyFAI quirk).
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("full", "csr", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # 2D no-split CSR: HistoBBox2d with delta=None -> each pixel
                # centre to one (radial, azimuthal) cell, coef 1.0. Same apply
                # (CsrIntegrator.integrate_ng) as 2D bbox-CSR. chiDiscAtPi default
                # True (not forwarded), pos1_period = CHI_DEG.period (clip flag).
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("no", "csr", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # Direct-split bbox histogram (splitBBox.histoBBox1d_engine): same
                # bbox boundaries + per-pixel overlap fractions as bbox-CSR, but
                # accumulated directly into bins (no sparse matrix). Differs from
                # bbox-CSR: the split coef delta_left/right cast (bin+1) to f32
                # (<float>) and the coef stays f64 into update_1d_accumulator
                # (no error-model fork: sum_nrm2 += (weight*norm)² in f64). Reduces
                # like the 2D/CSR path (guard on count, f64 intensity/sem/std,
                # f64 sums) — NOT like the no-split f32 histogram. Own golden.
                "npt": 1000,
                "unit": "q_nm^-1",
                "method": ("bbox", "histogram", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # 2D direct-split bbox histogram (splitBBox.histoBBox2d_engine):
                # 4-branch bbox overlap (radial × azimuthal) accumulated via
                # update_2d_accumulator (norm² = (norm·norm f32)·weight²). Unlike
                # the 1D engine, the delta casts use f64 (<position_t>). Reduction
                # = histogram2d (count guard, transpose to (azim, rad)). chiDiscAtPi
                # default True; pos1_period = CHI_DEG.period (clip flag).
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("bbox", "histogram", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # Full direct-split histogram (splitPixel.fullSplit1D_engine): the
                # same corner clipping as full-CSR (recenter + _integrate1d over the
                # 4 edges → per-bin trapezoid buffer), but the normalized overlap
                # buffer[bin]*inv_area is scattered straight into update_1d_accumulator
                # (f64 coef, no error-model fork) instead of stored as an f32 CSR
                # coef. Reduces like the CSR / 2D-histogram path (guard on count,
                # f64 sums). Own golden (reduction ORDER differs from full-CSR).
                "npt": 1000,
                "unit": "q_nm^-1",
                "method": ("full", "histogram", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
            {
                # 2D full direct-split histogram (splitPixel.fullSplit2D_engine):
                # corner clipping into a (w0+1)×(w1+1) box via _integrate2d (the
                # fused-type _calc_area float/double resolution shared with full-CSR
                # build), normalized box[i,j]*inv_area scattered via
                # update_2d_accumulator. Range skip uses min1>pos1_maxin (NOT the
                # full-CSR build's min1>=pos1_max). chiDiscAtPi default True,
                # pos1_period = CHI_DEG.period (360, applied to radian azimuths).
                "dim": 2,
                "npt_rad": 100,
                "npt_azim": 36,
                "unit": "q_nm^-1",
                "method": ("full", "histogram", "cython"),
                "error_model": "poisson",
                "correct_solid_angle": True,
                "polarization_factor": 0.99,
            },
        ],
    )
    print("done.")


if __name__ == "__main__":
    main()
