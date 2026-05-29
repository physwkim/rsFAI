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
    cython golden).  ``opencl`` falls back to pyFAI so a GUI GPU selection is not
    silently downgraded to rsFAI's CPU path.
  * no ``radial_range`` / ``azimuth_range`` is given (the range-override
    orchestration is not reimplemented here — it falls back to pyFAI).

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
        """Run pyFAI's own input normalization, then the rsFAI preproc.

        Returns a context dict the per-dimension dispatchers consume. Raises
        ``_Unsupported`` for any configuration outside rsFAI's coverage so the
        caller can fall back to pyFAI.
        """
        method = self._normalize_method(
            kwargs.get("method", ("bbox", "csr", "cython")),
            dim=dim,
            default=self.DEFAULT_METHOD_1D if dim == 1 else self.DEFAULT_METHOD_2D,
        )
        if method.dimension != dim:
            raise _Unsupported(f"method dimension {method.dimension} != {dim}")
        split, algo, impl = method.split_lower, method.algo_lower, method.impl_lower
        if impl not in _IMPLS:
            raise _Unsupported(f"impl {impl!r} (only {sorted(_IMPLS)} routed to rsFAI)")
        if not _supports(split, algo, dim):
            raise _Unsupported(f"split/algo ({split},{algo}) for dim={dim}")
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

        return {
            "method": method,
            "split": split,
            "algo": algo,
            "unit": unit,
            "shape": shape,
            "mask_i8": mask_i8,
            "prep": prep,
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

    # ---- 1D dispatch ------------------------------------------------------
    def _rsfai_integrate1d(self, data, npt, kwargs):
        ctx = self._rsfai_common(data, 1, kwargs)
        split, algo, unit = ctx["split"], ctx["algo"], ctx["unit"]
        prep, mask_i8, em, empty = ctx["prep"], ctx["mask_i8"], ctx["em"], ctx["empty"]
        allow_neg, shape = ctx["allow_neg"], ctx["shape"]

        if algo in ("csr", "csc"):
            if split in ("no", "bbox"):
                pos0 = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                dpos0 = (
                    np.ascontiguousarray(self.delta_array(shape, unit=unit).reshape(-1))
                    if split == "bbox"
                    else None
                )
                build = rsfai.build_bbox_csr_1d if algo == "csr" else rsfai.build_bbox_csc_1d
                data_m, indices, indptr, bc = build(
                    pos0, delta_pos0=dpos0, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg
                )
            else:  # full
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                build = rsfai.build_full_csr_1d if algo == "csr" else rsfai.build_full_csc_1d
                data_m, indices, indptr, bc = build(
                    corners, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, pos1_period=_TWO_PI,
                )
            integ = rsfai.csr_integrate1d if algo == "csr" else rsfai.csc_integrate1d
            out = integ(data_m, indices, indptr, prep, bc, error_model=em, empty=empty)
            position = bc

        elif algo == "lut":
            if split in ("no", "bbox"):
                pos0 = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                dpos0 = (
                    np.ascontiguousarray(self.delta_array(shape, unit=unit).reshape(-1))
                    if split == "bbox"
                    else None
                )
                idx, coef, lut_size, bc = rsfai.build_bbox_lut_1d(
                    pos0, delta_pos0=dpos0, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg
                )
            else:  # full
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                idx, coef, lut_size, bc = rsfai.build_full_lut_1d(
                    corners, mask=mask_i8, bins=npt, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, pos1_period=_TWO_PI,
                )
            out = rsfai.lut_integrate1d(idx, coef, lut_size, prep, bc, error_model=em, empty=empty)
            position = bc

        else:  # histogram
            if split == "no":
                radial = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                out = rsfai.histogram1d(radial, prep, npt, error_model=em, empty=empty)
            elif split == "bbox":
                radial = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                dpos0 = np.ascontiguousarray(self.delta_array(shape, unit=unit).reshape(-1))
                out = rsfai.histogram1d_bbox(
                    radial, dpos0, prep, mask=mask_i8, npt=npt, error_model=em,
                    empty=empty, allow_pos0_neg=allow_neg,
                )
            else:  # full
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                out = rsfai.histogram1d_full(
                    corners, prep, mask=mask_i8, npt=npt, error_model=em, empty=empty,
                    allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=_TWO_PI,
                )
            position = out["position"]

        radial = np.ascontiguousarray(position, dtype=np.float64) * unit.scale
        return self._build_1d_result(out, radial, ctx)

    # ---- 2D dispatch ------------------------------------------------------
    def _rsfai_integrate2d(self, data, npt_rad, npt_azim, kwargs):
        ctx = self._rsfai_common(data, 2, kwargs)
        split, algo, unit = ctx["split"], ctx["algo"], ctx["unit"]
        prep, mask_i8, em, empty = ctx["prep"], ctx["mask_i8"], ctx["em"], ctx["empty"]
        allow_neg, shape = ctx["allow_neg"], ctx["shape"]
        bins = (npt_rad, npt_azim)
        # 2D azimuthal binning is in chi radians; the reported axis is chi_deg.
        # pyFAI applies pos1_period = CHI_DEG.period (360) to the radian azimuth
        # with chiDiscAtPi=True for every 2D engine (see gen_golden manifest).
        period = float(units.CHI_DEG.period)

        if algo in ("csr", "csc"):
            if split in ("no", "bbox"):
                pos0 = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                pos1 = np.ascontiguousarray(
                    self.center_array(shape, unit="chi_rad", scale=False).reshape(-1)
                )
                dpos0 = dpos1 = None
                if split == "bbox":
                    dpos0 = np.ascontiguousarray(self.delta_array(shape, unit=unit).reshape(-1))
                    dpos1 = np.ascontiguousarray(self.delta_array(shape, unit="chi_rad").reshape(-1))
                build = rsfai.build_bbox_csr_2d if algo == "csr" else rsfai.build_bbox_csc_2d
                data_m, indices, indptr, bc0, bc1 = build(
                    pos0, pos1, delta_pos0=dpos0, delta_pos1=dpos1, mask=mask_i8, bins=bins,
                    allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=period,
                )
            else:  # full
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                build = rsfai.build_full_csr_2d if algo == "csr" else rsfai.build_full_csc_2d
                data_m, indices, indptr, bc0, bc1 = build(
                    corners, mask=mask_i8, bins=bins, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, pos1_period=period,
                )
            integ = rsfai.csr_integrate2d if algo == "csr" else rsfai.csc_integrate2d
            out = integ(data_m, indices, indptr, prep, bc0, bc1, error_model=em, empty=empty)

        elif algo == "lut":
            if split in ("no", "bbox"):
                pos0 = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                pos1 = np.ascontiguousarray(
                    self.center_array(shape, unit="chi_rad", scale=False).reshape(-1)
                )
                dpos0 = dpos1 = None
                if split == "bbox":
                    dpos0 = np.ascontiguousarray(self.delta_array(shape, unit=unit).reshape(-1))
                    dpos1 = np.ascontiguousarray(self.delta_array(shape, unit="chi_rad").reshape(-1))
                idx, coef, lut_size, bc0, bc1 = rsfai.build_bbox_lut_2d(
                    pos0, pos1, delta_pos0=dpos0, delta_pos1=dpos1, mask=mask_i8, bins=bins,
                    allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=period,
                )
            else:  # full
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                idx, coef, lut_size, bc0, bc1 = rsfai.build_full_lut_2d(
                    corners, mask=mask_i8, bins=bins, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, pos1_period=period,
                )
            out = rsfai.lut_integrate2d(idx, coef, lut_size, prep, bc0, bc1, error_model=em, empty=empty)

        else:  # histogram
            if split == "no":
                radial = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                azimuthal = np.ascontiguousarray(
                    self.center_array(shape, unit="chi_rad", scale=False).reshape(-1)
                )
                out = rsfai.histogram2d(
                    radial, azimuthal, prep, bins=bins, mask=mask_i8, error_model=em,
                    allow_radial_neg=allow_neg, chi_disc_at_pi=True, pos1_period=period, empty=empty,
                )
            elif split == "bbox":
                radial = np.ascontiguousarray(
                    self.center_array(shape, unit=unit, scale=False).reshape(-1)
                )
                dpos0 = np.ascontiguousarray(self.delta_array(shape, unit=unit).reshape(-1))
                azimuthal = np.ascontiguousarray(
                    self.center_array(shape, unit="chi_rad", scale=False).reshape(-1)
                )
                dpos1 = np.ascontiguousarray(self.delta_array(shape, unit="chi_rad").reshape(-1))
                out = rsfai.histogram2d_bbox(
                    radial, dpos0, azimuthal, dpos1, prep, bins=bins, mask=mask_i8,
                    allow_pos0_neg=allow_neg, chi_disc_at_pi=True, pos1_period=period,
                    error_model=em, empty=empty,
                )
            elif split == "full":
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                out = rsfai.histogram2d_full(
                    corners, prep, bins=bins, mask=mask_i8, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, pos1_period=period, error_model=em, empty=empty,
                )
            else:  # pseudo (2D only); the engine forwards no pos1_period
                corners = np.ascontiguousarray(
                    self.corner_array(shape, unit=unit, scale=False).astype(np.float64).reshape(-1)
                )
                out = rsfai.histogram2d_pseudo(
                    corners, prep, bins=bins, mask=mask_i8, allow_pos0_neg=allow_neg,
                    chi_disc_at_pi=True, error_model=em, empty=empty,
                )

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
