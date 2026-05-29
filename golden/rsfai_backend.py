#!/usr/bin/env python3
"""Use rsFAI's Rust integration kernels as the backend of a pyFAI
``AzimuthalIntegrator`` — a drop-in for the pyFAI-calibrate GUI and any code
that calls ``integrate1d``/``integrate2d``.

Motivation
----------
The GUI (``pyFAI.gui.tasks.IntegrationTask``) drives integration through a plain
``AzimuthalIntegrator.integrate1d_ng`` / ``integrate2d`` call.  To put rsFAI
*behind* that call without forking the GUI, we subclass the integrator and
override only the integration entry points; everything else (geometry, the PONI,
detector model, ``tth``/``chi``/``center_array``, ``getFit2D``, parallax,
``GeometryRefinement``) is inherited from pyFAI unchanged.

Why a Python subclass and not the Rust ``rsfai.AzimuthalIntegrator`` drop-in
----------------------------------------------------------------------------
The Rust high-level drop-in (``rsfai.AzimuthalIntegrator.load(poni)``) regenerates
positions/corrections in Rust and so is limited to the detectors rsFAI has
ported and takes no runtime mask/dark/flat.  The GUI hands integration an
*arbitrary* detector and a runtime calibration **mask**.  This subclass sidesteps
both limits: the per-pixel geometry arrays (``center_array``/``delta_array``/
``corner_array``) and the correction/error-model normalization come from the
inherited pyFAI methods — so **any** pyFAI detector works and the runtime
mask/dark/flat are honoured — while the heavy per-frame compute (preproc + bin +
reduce) runs in rsFAI.  This is exactly the recipe ``golden/gen_golden.py``
dumps and ``golden/test_inprocess_parity.py`` validates bit-for-bit, applied to a
live geometry instead of a committed dataset.

Supported / fallback
---------------------
A call dispatches to rsFAI when, after pyFAI's own method normalization:

  * ``split`` ∈ {no, bbox, full} and ``algo`` ∈ {csr, csc, lut, histogram}
    (1D), plus the 2D-only ``(pseudo, histogram)``;
  * the implementation is ``cython`` or ``python`` (both CPU; bit-exact vs the
    cython golden), or ``opencl`` with ``algo`` ∈ {csr, lut} — routed to
    rsFAI's ``GpuEngine``, which applies the corrections and the reduce
    on-device and reproduces pyFAI's OpenCL CSR/LUT within float-reduction
    tolerance (bit-exact in practice on this device).  The GPU path needs an
    *integer* image (the engine value-casts int32->float on-device); a float
    image falls back to pyFAI.  ``opencl`` with csc/histogram also falls back
    (no GPU csc engine; the GPU histogram needs radial bounds this backend
    does not derive).
  * no ``radial_range`` / ``azimuth_range`` is given (the range-override
    orchestration is not reimplemented here — it falls back to pyFAI).

The GPU engine is rebuilt on every call (the calibration GUI changes geometry
between integrations, so a cached engine would be stale); streaming a fixed
geometry over many frames should use ``rsfai.GpuEngine`` directly to amortise
the matrix upload and the kernel compile.

Anything else falls back to ``super()``.  Fallback is triggered only by the
``_Unsupported`` sentinel; a genuine error inside the rsFAI path propagates (it
is a bug, not an unsupported config) rather than being masked by a pyFAI result.

Result fidelity
---------------
The returned ``Integrate1dResult`` / ``Integrate2dResult`` is populated with the
same fields pyFAI sets (radial/azimuthal axes, intensity, and — when the error
model requests variance — ``sigma``/``sum_variance``/``std``/``sem``/
``sum_normalization2``), so existing result consumers are unaffected.  The
variance-family fields are gated on ``error_model.do_variance`` uniformly across
engines (matching pyFAI's histogram path; the GUI's default ``error_model=None``
sets none of them and reads only radial+intensity).

Usage
-----
    import rsfai_backend
    ai = rsfai_backend.RsfaiAzimuthalIntegrator(...)            # construct directly
    # or wrap an existing integrator (any AzimuthalIntegrator subclass instance,
    # GeometryRefinement included), keeping its identity and extra methods:
    ai = pyFAI.load("geometry.poni")
    rsfai_backend.install(ai)
    res = ai.integrate1d_ng(data, 1000, unit="q_nm^-1", method=("bbox","csr","cython"))
"""

import logging
import math

import numpy as np

from pyFAI import units
from pyFAI.containers import ErrorModel, Integrate1dResult, Integrate2dResult
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

import rsfai

_logger = logging.getLogger(__name__)

# Splits/algos rsFAI's kernels cover (mirrors golden/test_inprocess_parity.py).
_SPLITS_1D = frozenset(("no", "bbox", "full"))
_ALGOS = frozenset(("csr", "csc", "lut", "histogram"))
# Implementations routed to rsFAI's CPU kernels. ``opencl`` is left to pyFAI.
_IMPLS = frozenset(("cython", "python"))
_TWO_PI = 2.0 * math.pi


class _Unsupported(Exception):
    """Raised by the rsFAI path to request fallback to pyFAI for this call."""


def _supports(split, algo, dim):
    if algo not in _ALGOS:
        return False
    if split in _SPLITS_1D:
        return True
    if split == "pseudo":
        return dim == 2 and algo == "histogram"
    return False


def _f32(arr):
    """Contiguous flat float32 view, or None."""
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


def _get(out, *names):
    """First present key among rsFAI's inconsistent dict layouts.

    The no-split histogram and the 2D dicts key the binned sums as
    ``signal``/``variance``/``normalization``/``norm_sq``; the split histograms
    and the sparse engines key them ``sum_signal``/``sum_variance``/
    ``sum_normalization``/``sum_norm_sq``.
    """
    for n in names:
        if n in out:
            return out[n]
    raise KeyError(f"none of {names} in rsFAI output (keys={list(out)})")


def _gpu_corr_kwargs(corr):
    """``GpuEngine`` constructor kwargs from a ``ctx['gpu_corr']`` bundle.

    Absent correction arrays are dropped so the engine's ``do_*`` flags stay off
    (it derives them from array presence), and a ``dummy`` of ``None`` is omitted
    so dummy-masking stays disabled.
    """
    kw = {
        "error_model": corr["error_model"],
        "delta_dummy": corr["delta_dummy"],
        "normalization_factor": corr["normalization_factor"],
        # NG integration keeps signal and norm separate (the reduce divides);
        # mirror the CPU preproc4(apply_normalization=False) path and gen_golden.
        "apply_normalization": False,
        "empty": corr["empty"],
        "prefer_gpu": True,
        "mask": corr["mask"],
    }
    if corr["dummy"] is not None:
        kw["dummy"] = corr["dummy"]
    for name in ("variance", "dark", "flat", "solidangle", "polarization", "absorption"):
        if corr[name] is not None:
            kw[name] = corr[name]
    return kw


class RsfaiBackendMixin:
    """Overrides ``integrate1d``/``integrate2d`` (and their ``_ng`` aliases) to
    run on rsFAI, falling back to the inherited pyFAI implementation.

    Mixed in *before* a pyFAI ``AzimuthalIntegrator`` (or a subclass such as
    ``GeometryRefinement``) so ``super()`` reaches the original implementation.
    """

    # ---- 1D ---------------------------------------------------------------
    def integrate1d(self, data, npt, **kwargs):
        try:
            return self._rsfai_integrate1d(data, npt, kwargs)
        except _Unsupported as exc:
            _logger.debug("rsFAI 1D backend -> pyFAI fallback: %s", exc)
            return super().integrate1d(data, npt, **kwargs)

    # pyFAI binds ``integrate1d_ng = _integrate1d_ng = integrate1d`` at class
    # scope (azimuthal.py); rebind the aliases here so the GUI's ``_ng`` entry
    # point hits the override and not the inherited base method object.
    integrate1d_ng = integrate1d
    _integrate1d_ng = integrate1d

    # ---- 2D ---------------------------------------------------------------
    def integrate2d(self, data, npt_rad, npt_azim=360, **kwargs):
        try:
            return self._rsfai_integrate2d(data, npt_rad, npt_azim, kwargs)
        except _Unsupported as exc:
            _logger.debug("rsFAI 2D backend -> pyFAI fallback: %s", exc)
            return super().integrate2d(data, npt_rad, npt_azim, **kwargs)

    integrate2d_ng = integrate2d
    _integrate2d_ng = integrate2d

    # ---- shared normalization --------------------------------------------
    def _rsfai_common(self, data, dim, kwargs):
        """Run pyFAI's own input normalization, then prepare the rsFAI inputs.

        Returns a context dict the per-dimension dispatchers consume. For the CPU
        path the per-pixel preproc is run here (``ctx['prep']``); for the GPU
        path the corrections are carried raw (``ctx['gpu_corr']`` +
        ``ctx['image_i32']``) because rsFAI's ``GpuEngine`` applies them
        on-device from the raw image.  Raises ``_Unsupported`` for any
        configuration outside rsFAI's coverage so the caller can fall back.
        """
        method = self._normalize_method(
            kwargs.get("method", ("bbox", "csr", "cython")),
            dim=dim,
            default=self.DEFAULT_METHOD_1D if dim == 1 else self.DEFAULT_METHOD_2D,
        )
        if method.dimension != dim:
            raise _Unsupported(f"method dimension {method.dimension} != {dim}")
        split, algo, impl = method.split_lower, method.algo_lower, method.impl_lower
        if not _supports(split, algo, dim):
            raise _Unsupported(f"split/algo ({split},{algo}) for dim={dim}")
        if impl == "opencl":
            # rsFAI's GpuEngine covers CSR/LUT (sparse-matrix reduce on-device).
            # There is no GPU csc engine, and the GPU histogram needs radial
            # bounds this backend does not derive — leave both to pyFAI's OpenCL.
            gpu = True
            if algo not in ("csr", "lut"):
                raise _Unsupported(f"opencl {algo} (GPU path covers csr/lut only)")
        elif impl in _IMPLS:
            gpu = False
        else:
            raise _Unsupported(f"impl {impl!r} not routed to rsFAI")
        if kwargs.get("radial_range") is not None or kwargs.get("azimuth_range") is not None:
            raise _Unsupported("radial_range/azimuth_range override not ported")

        unit = units.to_unit(kwargs.get("unit", units.Q))
        shape = data.shape

        dummy, delta_dummy = self._normalize_dummies(
            kwargs.get("dummy"), kwargs.get("delta_dummy"), data
        )
        # The combined static mask (detector mask + user mask), canonicalized by
        # create_mask (False=valid, True=bad) — the same array gen_golden dumps.
        mask_i8 = np.ascontiguousarray(
            self.create_mask(data, mask=kwargs.get("mask")).astype(np.int8)
        ).reshape(-1)
        _, _, has_mask = self._normalize_mask(kwargs.get("mask"))

        correct_solid_angle = kwargs.get("correctSolidAngle", True)
        solidangle, _ = self._normalize_solidangle(shape, correct_solid_angle, with_checksum=False)
        polarization_factor = kwargs.get("polarization_factor")
        polarization, _ = self._normalize_polarization(shape, polarization_factor, with_checksum=False)
        dark, has_dark = self._normalize_dark(kwargs.get("dark"))
        flat, has_flat = self._normalize_flat(kwargs.get("flat"))
        absorption = kwargs.get("absorption")

        error_model, variance = self._normalize_error_model_variance(
            data, method, dark, kwargs.get("error_model"), kwargs.get("variance")
        )
        if error_model == ErrorModel.VARIANCE and variance is None:
            raise _Unsupported("variance error model without a variance array")

        normalization_factor = float(kwargs.get("normalization_factor", 1.0))
        empty = float(self._empty)

        if gpu:
            # GpuEngine does dark/flat/solidangle/polarization/absorption,
            # dummy-masking and the (Poisson/variance) error model on-device from
            # the raw image, so no host-side preproc4 is run. rsFAI's GpuEngine
            # integrates an integer image with value semantics: the LUT/histogram
            # kernels host-cast int32->float32 (pyFAI's s32_to_float) and CSR reads
            # it with dtype-code -4 (int32). A non-integer image cannot be carried
            # losslessly that way, so it falls back to pyFAI's OpenCL path.
            if not np.issubdtype(np.asarray(data).dtype, np.integer):
                raise _Unsupported(
                    "opencl path needs an integer image (GpuEngine value-casts "
                    "int32->float); float image -> pyFAI fallback"
                )
            prep = None
            image_i32 = np.ascontiguousarray(data, dtype=np.int32).reshape(-1)
            gpu_corr = {
                "variance": _f32(variance),
                "dark": _f32(dark),
                "flat": _f32(flat),
                "solidangle": _f32(solidangle),
                "polarization": _f32(polarization),
                "absorption": _f32(absorption),
                "mask": mask_i8,
                "error_model": int(error_model),
                "dummy": float(dummy) if dummy is not None else None,
                "delta_dummy": float(delta_dummy) if delta_dummy is not None else 0.0,
                "normalization_factor": normalization_factor,
                "empty": empty,
            }
        else:
            prep = rsfai.preproc4(
                np.ascontiguousarray(data, dtype=np.float32).reshape(-1),
                dark=_f32(dark),
                flat=_f32(flat),
                solidangle=_f32(solidangle),
                polarization=_f32(polarization),
                absorption=_f32(absorption),
                mask=mask_i8,
                variance=_f32(variance),
                normalization_factor=normalization_factor,
                poissonian=bool(error_model.poissonian),
                check_dummy=dummy is not None,
                dummy=float(dummy) if dummy is not None else 0.0,
                delta_dummy=float(delta_dummy) if delta_dummy is not None else 0.0,
                # The NG integration keeps signal and norm separate (the reduce does
                # sum_signal/sum_normalization); pre-dividing is the legacy path.
                # Matches pyFAI ext.preproc.preproc's default and gen_golden.
                apply_normalization=False,
            )
            image_i32 = None
            gpu_corr = None

        return {
            "method": method,
            "split": split,
            "algo": algo,
            "unit": unit,
            "shape": shape,
            "mask_i8": mask_i8,
            "prep": prep,
            "gpu": gpu,
            "gpu_corr": gpu_corr,
            "image_i32": image_i32,
            "em": int(error_model),
            "error_model": error_model,
            "allow_neg": not unit.positive,
            "empty": empty,
            "has_mask": has_mask,
            "has_dark": has_dark,
            "has_flat": has_flat,
            "polarization_factor": polarization_factor,
            "normalization_factor": normalization_factor,
            "metadata": kwargs.get("metadata"),
        }

    # ---- geometry arrays (shared by CPU + GPU, sparse + histogram) --------
    def _center0(self, ctx):  # radial bin variable (the requested unit)
        return np.ascontiguousarray(
            self.center_array(ctx["shape"], unit=ctx["unit"], scale=False).reshape(-1)
        )

    def _delta0(self, ctx):
        return np.ascontiguousarray(
            self.delta_array(ctx["shape"], unit=ctx["unit"]).reshape(-1)
        )

    def _corners(self, ctx):
        return np.ascontiguousarray(
            self.corner_array(ctx["shape"], unit=ctx["unit"], scale=False)
            .astype(np.float64).reshape(-1)
        )

    def _center1(self, ctx):  # azimuth in chi radians (2D)
        return np.ascontiguousarray(
            self.center_array(ctx["shape"], unit="chi_rad", scale=False).reshape(-1)
        )

    def _delta1(self, ctx):
        return np.ascontiguousarray(
            self.delta_array(ctx["shape"], unit="chi_rad").reshape(-1)
        )

    # ---- sparse matrix build (shared by the CPU kernels + the GPU engine) -
    def _build_sparse_1d(self, ctx, npt):
        """Build the 1D CSR/CSC/LUT matrix from the inherited geometry.

        Returns ``(kind, payload, bin_centers)`` with ``payload`` =
        ``(data, indices, indptr)`` for csr/csc or ``(idx, coef, lut_size)`` for
        lut, so the CPU integrate kernels and the GPU engine share one build.
        """
        split, algo = ctx["split"], ctx["algo"]
        mask_i8, allow_neg = ctx["mask_i8"], ctx["allow_neg"]
        if split in ("no", "bbox"):
            pos0 = self._center0(ctx)
            dpos0 = self._delta0(ctx) if split == "bbox" else None
            if algo in ("csr", "csc"):
                build = rsfai.build_bbox_csr_1d if algo == "csr" else rsfai.build_bbox_csc_1d
                data_m, indices, indptr, bc = build(
                    pos0, delta_pos0=dpos0, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg
                )
                return algo, (data_m, indices, indptr), bc
            idx, coef, lut_size, bc = rsfai.build_bbox_lut_1d(
                pos0, delta_pos0=dpos0, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg
            )
            return "lut", (idx, coef, lut_size), bc
        # full
        corners = self._corners(ctx)
        if algo in ("csr", "csc"):
            build = rsfai.build_full_csr_1d if algo == "csr" else rsfai.build_full_csc_1d
            data_m, indices, indptr, bc = build(
                corners, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg,
                chi_disc_at_pi=True, pos1_period=_TWO_PI,
            )
            return algo, (data_m, indices, indptr), bc
        idx, coef, lut_size, bc = rsfai.build_full_lut_1d(
            corners, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg,
            chi_disc_at_pi=True, pos1_period=_TWO_PI,
        )
        return "lut", (idx, coef, lut_size), bc

    def _build_sparse_2d(self, ctx, bins, period):
        """Build the 2D CSR/CSC/LUT matrix; returns ``(kind, payload, bc0, bc1)``."""
        split, algo = ctx["split"], ctx["algo"]
        mask_i8, allow_neg = ctx["mask_i8"], ctx["allow_neg"]
        if split in ("no", "bbox"):
            pos0 = self._center0(ctx)
            pos1 = self._center1(ctx)
            dpos0 = self._delta0(ctx) if split == "bbox" else None
            dpos1 = self._delta1(ctx) if split == "bbox" else None
            if algo in ("csr", "csc"):
                build = rsfai.build_bbox_csr_2d if algo == "csr" else rsfai.build_bbox_csc_2d
                data_m, indices, indptr, bc0, bc1 = build(
                    pos0, pos1, delta_pos0=dpos0, delta_pos1=dpos1, mask=mask_i8, bins=bins,
                    allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=period,
                )
                return algo, (data_m, indices, indptr), bc0, bc1
            idx, coef, lut_size, bc0, bc1 = rsfai.build_bbox_lut_2d(
                pos0, pos1, delta_pos0=dpos0, delta_pos1=dpos1, mask=mask_i8, bins=bins,
                allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=period,
            )
            return "lut", (idx, coef, lut_size), bc0, bc1
        # full
        corners = self._corners(ctx)
        if algo in ("csr", "csc"):
            build = rsfai.build_full_csr_2d if algo == "csr" else rsfai.build_full_csc_2d
            data_m, indices, indptr, bc0, bc1 = build(
                corners, mask=mask_i8, bins=bins, allow_pos0_neg=allow_neg,
                chi_disc_at_pi=True, pos1_period=period,
            )
            return algo, (data_m, indices, indptr), bc0, bc1
        idx, coef, lut_size, bc0, bc1 = rsfai.build_full_lut_2d(
            corners, mask=mask_i8, bins=bins, allow_pos0_neg=allow_neg,
            chi_disc_at_pi=True, pos1_period=period,
        )
        return "lut", (idx, coef, lut_size), bc0, bc1

    # ---- 1D dispatch ------------------------------------------------------
    def _rsfai_integrate1d(self, data, npt, kwargs):
        ctx = self._rsfai_common(data, 1, kwargs)
        if ctx["gpu"]:
            return self._rsfai_gpu_integrate1d(ctx, npt)
        algo, unit = ctx["algo"], ctx["unit"]
        prep, mask_i8, em, empty = ctx["prep"], ctx["mask_i8"], ctx["em"], ctx["empty"]
        allow_neg, split = ctx["allow_neg"], ctx["split"]

        if algo == "histogram":
            if split == "no":
                out = rsfai.histogram1d(self._center0(ctx), prep, npt, error_model=em, empty=empty)
            elif split == "bbox":
                out = rsfai.histogram1d_bbox(
                    self._center0(ctx), self._delta0(ctx), prep, mask=mask_i8, npt=npt,
                    error_model=em, empty=empty, allow_pos0_neg=allow_neg,
                )
            else:  # full
                out = rsfai.histogram1d_full(
                    self._corners(ctx), prep, mask=mask_i8, npt=npt, error_model=em, empty=empty,
                    allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=_TWO_PI,
                )
            position = out["position"]
        else:
            kind, payload, position = self._build_sparse_1d(ctx, npt)
            if kind in ("csr", "csc"):
                data_m, indices, indptr = payload
                integ = rsfai.csr_integrate1d if kind == "csr" else rsfai.csc_integrate1d
                out = integ(data_m, indices, indptr, prep, position, error_model=em, empty=empty)
            else:  # lut
                idx, coef, lut_size = payload
                out = rsfai.lut_integrate1d(
                    idx, coef, lut_size, prep, position, error_model=em, empty=empty
                )

        radial = np.ascontiguousarray(position, dtype=np.float64) * unit.scale
        return self._build_1d_result(out, radial, ctx)

    # ---- 2D dispatch ------------------------------------------------------
    def _rsfai_integrate2d(self, data, npt_rad, npt_azim, kwargs):
        ctx = self._rsfai_common(data, 2, kwargs)
        algo = ctx["algo"]
        prep, mask_i8, em, empty = ctx["prep"], ctx["mask_i8"], ctx["em"], ctx["empty"]
        allow_neg, split = ctx["allow_neg"], ctx["split"]
        bins = (npt_rad, npt_azim)
        # 2D azimuthal binning is in chi radians; the reported axis is chi_deg.
        # pyFAI applies pos1_period = CHI_DEG.period (360) to the radian azimuth
        # with chiDiscAtPi=True for every 2D engine (see gen_golden manifest).
        period = float(units.CHI_DEG.period)

        if ctx["gpu"]:
            return self._rsfai_gpu_integrate2d(ctx, npt_rad, npt_azim, period)

        if algo == "histogram":
            if split == "no":
                out = rsfai.histogram2d(
                    self._center0(ctx), self._center1(ctx), prep, bins=bins, mask=mask_i8,
                    error_model=em, allow_radial_neg=allow_neg, chi_disc_at_pi=True,
                    pos1_period=period, empty=empty,
                )
            elif split == "bbox":
                out = rsfai.histogram2d_bbox(
                    self._center0(ctx), self._delta0(ctx), self._center1(ctx), self._delta1(ctx),
                    prep, bins=bins, mask=mask_i8, allow_pos0_neg=allow_neg, chi_disc_at_pi=True,
                    pos1_period=period, error_model=em, empty=empty,
                )
            elif split == "full":
                out = rsfai.histogram2d_full(
                    self._corners(ctx), prep, bins=bins, mask=mask_i8, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, pos1_period=period, error_model=em, empty=empty,
                )
            else:  # pseudo (2D only); the engine forwards no pos1_period
                out = rsfai.histogram2d_pseudo(
                    self._corners(ctx), prep, bins=bins, mask=mask_i8, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, error_model=em, empty=empty,
                )
        else:
            kind, payload, bc0, bc1 = self._build_sparse_2d(ctx, bins, period)
            if kind in ("csr", "csc"):
                data_m, indices, indptr = payload
                integ = rsfai.csr_integrate2d if kind == "csr" else rsfai.csc_integrate2d
                out = integ(data_m, indices, indptr, prep, bc0, bc1, error_model=em, empty=empty)
            else:  # lut
                idx, coef, lut_size = payload
                out = rsfai.lut_integrate2d(
                    idx, coef, lut_size, prep, bc0, bc1, error_model=em, empty=empty
                )

        return self._build_2d_result(out, npt_rad, npt_azim, ctx)

    # ---- GPU dispatch (OpenCL CSR/LUT via rsFAI's GpuEngine) --------------
    def _rsfai_gpu_integrate1d(self, ctx, npt):
        kind, payload, bc = self._build_sparse_1d(ctx, npt)
        image, image_size = ctx["image_i32"], int(np.prod(ctx["shape"]))
        bc64 = np.ascontiguousarray(bc, dtype=np.float64)
        if kind == "csr":
            data_m, indices, indptr = payload
            eng = rsfai.GpuEngine.from_csr_1d(
                np.ascontiguousarray(data_m, dtype=np.float32),
                np.ascontiguousarray(indices, dtype=np.int32),
                np.ascontiguousarray(indptr, dtype=np.int32),
                bc64, image_size, dtype=-4, **_gpu_corr_kwargs(ctx["gpu_corr"]),
            )
        else:  # lut (no dtype: the kernel host-casts the int32 image to float)
            idx, coef, lut_size = payload
            eng = rsfai.GpuEngine.from_lut_1d(
                np.ascontiguousarray(idx, dtype=np.int32),
                np.ascontiguousarray(coef, dtype=np.float32),
                bc64, image_size, int(lut_size), **_gpu_corr_kwargs(ctx["gpu_corr"]),
            )
        out = eng.integrate1d(image)
        radial = bc64 * ctx["unit"].scale
        return self._build_1d_result(out, radial, ctx)

    def _rsfai_gpu_integrate2d(self, ctx, npt_rad, npt_azim, period):
        kind, payload, bc0, bc1 = self._build_sparse_2d(ctx, (npt_rad, npt_azim), period)
        image, image_size = ctx["image_i32"], int(np.prod(ctx["shape"]))
        bc0_64 = np.ascontiguousarray(bc0, dtype=np.float64)
        bc1_64 = np.ascontiguousarray(bc1, dtype=np.float64)
        if kind == "csr":
            data_m, indices, indptr = payload
            eng = rsfai.GpuEngine.from_csr_2d(
                np.ascontiguousarray(data_m, dtype=np.float32),
                np.ascontiguousarray(indices, dtype=np.int32),
                np.ascontiguousarray(indptr, dtype=np.int32),
                bc0_64, bc1_64, image_size, dtype=-4, **_gpu_corr_kwargs(ctx["gpu_corr"]),
            )
        else:  # lut (no dtype: the kernel host-casts the int32 image to float)
            idx, coef, lut_size = payload
            eng = rsfai.GpuEngine.from_lut_2d(
                np.ascontiguousarray(idx, dtype=np.int32),
                np.ascontiguousarray(coef, dtype=np.float32),
                bc0_64, bc1_64, image_size, int(lut_size), **_gpu_corr_kwargs(ctx["gpu_corr"]),
            )
        out = eng.integrate2d(image)
        return self._build_2d_result(out, npt_rad, npt_azim, ctx)

    # ---- result assembly --------------------------------------------------
    def _build_1d_result(self, out, radial, ctx):
        error_model = ctx["error_model"]
        do_variance = error_model.do_variance
        intensity = np.asarray(out["intensity"])
        if do_variance:
            result = Integrate1dResult(radial, intensity, np.asarray(out["sigma"]))
            result._set_sum_variance(np.asarray(_get(out, "sum_variance", "variance")))
            result._set_std(np.asarray(out["std"]))
            result._set_sem(np.asarray(out["sem"]))
            result._set_sum_normalization2(np.asarray(_get(out, "sum_norm_sq", "norm_sq")))
        else:
            result = Integrate1dResult(radial, intensity)
        result._set_compute_engine(f"rsfai:{ctx['method']}")
        result._set_unit(ctx["unit"])
        result._set_dummy(ctx["empty"])
        result._set_sum_signal(np.asarray(_get(out, "sum_signal", "signal")))
        result._set_sum_normalization(np.asarray(_get(out, "sum_normalization", "normalization")))
        result._set_count(np.asarray(out["count"]))
        self._set_common_result(result, ctx, "integrate1d_ng")
        return result

    def _build_2d_result(self, out, npt_rad, npt_azim, ctx):
        error_model = ctx["error_model"]
        do_variance = error_model.do_variance

        def grid(flat):
            return np.asarray(flat).reshape(npt_azim, npt_rad)

        bins_rad = np.asarray(out["radial"], dtype=np.float64) * ctx["unit"].scale
        bins_azim = np.asarray(out["azimuthal"], dtype=np.float64) * float(units.CHI_DEG.scale)
        intensity = grid(out["intensity"])
        sigma = grid(out["sigma"]) if do_variance else None

        result = Integrate2dResult(intensity, bins_rad, bins_azim, sigma)
        result._set_compute_engine(f"rsfai:{ctx['method']}")
        result._set_radial_unit(ctx["unit"])
        result._set_azimuthal_unit(units.CHI_DEG)
        result._set_dummy(ctx["empty"])
        result._set_sum_signal(grid(_get(out, "sum_signal", "signal")))
        result._set_sum_normalization(grid(_get(out, "sum_normalization", "normalization")))
        result._set_count(grid(out["count"]))
        if do_variance:
            result._set_sum_normalization2(grid(_get(out, "sum_norm_sq", "norm_sq")))
            result._set_sum_variance(grid(_get(out, "sum_variance", "variance")))
            result._set_std(grid(out["std"]))
            result._set_sem(grid(out["sem"]))
        self._set_common_result(result, ctx, "integrate2d")
        return result

    @staticmethod
    def _set_common_result(result, ctx, method_called):
        result._set_method(ctx["method"])
        result._set_method_called(method_called)
        result._set_error_model(ctx["error_model"])
        result._set_has_mask_applied(ctx["has_mask"])
        result._set_has_dark_correction(ctx["has_dark"])
        result._set_has_flat_correction(ctx["has_flat"])
        result._set_polarization_factor(ctx["polarization_factor"])
        result._set_normalization_factor(ctx["normalization_factor"])
        result._set_metadata(ctx["metadata"])


class RsfaiAzimuthalIntegrator(RsfaiBackendMixin, AzimuthalIntegrator):
    """A pyFAI ``AzimuthalIntegrator`` whose ``integrate1d``/``integrate2d``
    run on rsFAI's Rust kernels, falling back to pyFAI for unsupported configs.
    """


def install(ai):
    """Rebind ``ai`` to use the rsFAI backend, preserving its class identity.

    Works on any ``AzimuthalIntegrator`` (or subclass such as
    ``GeometryRefinement``) instance: a dynamic subclass mixing
    ``RsfaiBackendMixin`` ahead of ``ai``'s current class is created and
    assigned to ``ai.__class__``, so the rsFAI overrides win while every other
    inherited method (including the subclass-specific ones) is retained and
    ``super()`` still reaches the original implementation.

    :param ai: the integrator instance to patch (mutated in place)
    :return: ``ai`` (for chaining)
    """
    base = type(ai)
    if issubclass(base, RsfaiBackendMixin):
        return ai
    patched = type(f"Rsfai_{base.__name__}", (RsfaiBackendMixin, base), {})
    ai.__class__ = patched
    return ai
