#!/usr/bin/env python3
"""In-process parity for the high-level drop-in: ``rsfai.AzimuthalIntegrator``
vs ``pyFAI``, PONI + image in — nothing else.

Run in the ``daq`` conda env (which has both ``pyFAI`` and the maturin-built
``rsfai`` installed):

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/test_inprocess_dropin.py

This is the high-level analog of ``test_inprocess_parity.py``. Where that test
feeds the rsfai *kernels* the dumped Tier-A intermediates, here the drop-in
``rsfai.AzimuthalIntegrator.load(poni).integrate1d/2d(image, …)`` regenerates
pixel positions, corrections, gap mask, dummy, and preproc rows itself from the
geometry and the frame — exactly as ``pyFAI.load(poni).integrate1d_ng(...)``
does — so the only inputs are the ``.poni`` and the image. It covers every
cython method tuple ``(split, algo[, impl])`` with ``split ∈ {no, bbox, full}``
and ``algo ∈ {histogram, csr, lut, csc}``, for 1D and 2D, passing ``method`` to
the drop-in. The 2D ``pseudo`` split is not ported, so its dataset is skipped,
as are detectors the drop-in cannot resolve.

Each exposed output field is checked against BOTH the committed golden AND a
live in-process pyFAI run, so by transitivity ``rsfai == live pyFAI``. The gate
mirrors ``crates/rsfai/tests/dropin_golden.rs`` and ``doc/bit-exact-ladder.md``:

  * bin-center **axes** (``radial``, ``azimuthal``) derive from order-independent
    min/max + linspace — gated **bit-exact** on both legs;
  * **accumulator** fields: the serial engines (sparse ``csr``/``lut``/``csc``
    and the split ``bbox``/``full`` histograms) run in pixel-index order, so the
    rsfai leg is gated **bit-exact**; only the no-split histogram is rayon-
    parallel (non-deterministic f64 add order) and is gated at relative error
    ``<= REL_TOL`` (1e-6). The live leg, single-threaded against single-threaded
    golden, is the libm anchor and stays **bit-exact** for every engine.

Every comparison reports bit-exactness, max-ULP, and max relative error, so the
observed divergence (0 in practice for the Pilatus1M datasets, where each bin's
f64 sum spans <53 bits and is exact) is visible and tolerance is never silently
widened. Exit code is 0 only if every field of every dataset passes its gate.
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyFAI  # noqa: E402
import rsfai  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"

# Relative-error gate for the parallel-histogram accumulator fields (the rsfai
# leg). Matches REL_TOL in crates/rsfai/tests/dropin_golden.rs.
REL_TOL = 1e-6

# Output fields beyond the axes, common to 1D and 2D.
ACCUMULATOR_FIELDS = (
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


def load(d, name):
    return np.load(DATASETS / d / f"{name}.npy")


def _mono_key(bits_signed, sign_bit):
    """Total-order key for an IEEE bit pattern read as a signed int."""
    return sign_bit - bits_signed if bits_signed < 0 else bits_signed


def _ulp_scalar(x, y, is_f32):
    if is_f32:
        bx = struct.unpack("<i", struct.pack("<f", float(x)))[0]
        by = struct.unpack("<i", struct.pack("<f", float(y)))[0]
        sign = 0x80000000
    else:
        bx = struct.unpack("<q", struct.pack("<d", float(x)))[0]
        by = struct.unpack("<q", struct.pack("<d", float(y)))[0]
        sign = 0x8000000000000000
    return abs(_mono_key(bx, sign) - _mono_key(by, sign))


def _max_rel(a, g):
    """Max relative diff |a-g|/|g| and the count of one-sided NaNs.

    g==0 requires a==g (rel 0, else +inf); both-NaN counts as equal; exactly one
    NaN is a mismatch (rel +inf and counted). Matches CompareReport::within_rel.
    """
    a = a.ravel().astype(np.float64)
    g = g.ravel().astype(np.float64)
    if a.size == 0:
        return 0.0, 0
    both_nan = np.isnan(a) & np.isnan(g)
    one_nan = np.isnan(a) ^ np.isnan(g)
    n_nan = int(np.count_nonzero(one_nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.abs(a - g)
        rel = np.where(g != 0.0, diff / np.abs(g), np.where(a == g, 0.0, np.inf))
    rel = np.where(both_nan, 0.0, rel)
    rel = np.where(one_nan, np.inf, rel)
    return float(np.max(rel)), n_nan


def compare(actual, golden):
    """Compare ``actual`` vs ``golden``; return (bit_exact, max_rel, n_nan, detail).

    A dtype or shape mismatch is reported as a hard failure (bit_exact=False,
    max_rel=+inf). The caller selects the gate per field/leg from the returned
    facts.
    """
    a = np.ascontiguousarray(actual)
    g = np.ascontiguousarray(golden)
    if a.dtype != g.dtype:
        return False, float("inf"), 0, f"dtype {a.dtype} != golden {g.dtype}"
    if a.shape != g.shape:
        return False, float("inf"), 0, f"shape {a.shape} != golden {g.shape}"

    bit_exact = a.tobytes() == g.tobytes()
    if a.dtype not in (np.float32, np.float64):
        n = 0 if bit_exact else int(np.count_nonzero(a.ravel() != g.ravel()))
        detail = "bit-exact" if bit_exact else f"{n}/{a.size} differ (dtype {a.dtype})"
        return bit_exact, 0.0 if bit_exact else float("inf"), 0, detail

    max_rel, n_nan = _max_rel(a, g)
    if bit_exact:
        return True, 0.0, 0, "bit-exact"

    af, gf = a.ravel(), g.ravel()
    is_f32 = a.dtype == np.float32
    diff_idx = np.nonzero(af != gf)[0]
    max_ulp = 0
    for i in diff_idx:
        xv, yv = af[i], gf[i]
        if np.isnan(xv) or np.isnan(yv):
            continue
        max_ulp = max(max_ulp, _ulp_scalar(xv, yv, is_f32))
    detail = f"{diff_idx.size}/{af.size} differ, max_ulp={max_ulp}, max_rel={max_rel:.2e}"
    if n_nan:
        detail += f", {n_nan} NaN-bit"
    return False, max_rel, n_nan, detail


def method_of(cfg):
    m = tuple(cfg["method"])
    return m[0], m[1]  # (split, algo)


def run_rsfai_dropin(d, cfg):
    """Drive ``rsfai.AzimuthalIntegrator`` from the PONI + image alone.

    Returns the field-name -> array dict, or None if the drop-in cannot handle
    this dataset's detector (so the caller skips it with a visible note).
    """
    try:
        ai = rsfai.AzimuthalIntegrator.load(str(DATASETS / d / "geometry.poni"))
    except ValueError as e:
        print(f"  SKIP rsfai drop-in: {e}")
        return None

    # int image -> f32, exactly the cast pyFAI applies before the cython preproc
    # (and the contract rsfai::AzimuthalIntegrator::integrate* requires).
    img = np.ascontiguousarray(load(d, "image").astype(np.float32))
    unit = cfg["unit"]
    em = cfg["error_model_code"]
    common = dict(
        method=tuple(cfg["method"]),
        correct_solid_angle=cfg["correct_solid_angle"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"],
        error_model=em,
    )
    # A user radial_range (scaled unit) overrides the radial axis; pass it to the
    # drop-in exactly as run_live passes it to pyFAI, so a range dataset compares
    # range-vs-range. The manifest stores it as a [lo, hi] list or null; the PyO3
    # signature takes an Option<(f64, f64)>, which accepts a 2-list or None.
    # (azimuth_range is not yet wired into the drop-in, so it is not forwarded.)
    if cfg.get("radial_range") is not None:
        common["radial_range"] = tuple(cfg["radial_range"])

    if cfg.get("dim", 1) == 2:
        out = ai.integrate2d(img, cfg["npt_rad"], cfg["npt_azim"], unit, **common)
    else:
        out = ai.integrate1d(img, cfg["npt"], unit, **common)

    # The drop-in already returns scaled axes and pyFAI-keyed fields, and the 2D
    # accumulator arrays are shaped (npt_azim, npt_rad) — same layout as golden.
    fields = {"radial": np.asarray(out["radial"])}
    if "azimuthal" in out:
        fields["azimuthal"] = np.asarray(out["azimuthal"])
    for f in ACCUMULATOR_FIELDS:
        fields[f] = np.asarray(out[f])
    return fields


def run_live(d, cfg):
    """Re-run pyFAI's high-level integrator in-process; field-name -> array."""
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = load(d, "image")
    common = dict(
        unit=cfg["unit"],
        method=tuple(cfg["method"]),
        correctSolidAngle=cfg["correct_solid_angle"],
        error_model=cfg["error_model"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"],
        radial_range=cfg["radial_range"],
        azimuth_range=cfg["azimuth_range"],
    )
    if cfg.get("dim", 1) == 2:
        res = ai.integrate2d_ng(img, cfg["npt_rad"], cfg["npt_azim"], **common)
    else:
        res = ai.integrate1d_ng(img, cfg["npt"], **common)
    attrs = ("radial", "azimuthal") + ACCUMULATOR_FIELDS
    out = {}
    for a in attrs:
        v = getattr(res, a, None)
        if isinstance(v, np.ndarray):
            out[a] = v
    return out


def dataset_dirs():
    # Skip OpenCL datasets: they carry a reduced manifest (no error_model_code /
    # cython intermediates) and are validated by the rsfai-opencl harness, not
    # this cython drop-in test. Mirrors the Rust golden discovery filter.
    return sorted(
        p.name
        for p in DATASETS.iterdir()
        if (p / "manifest.json").exists() and not (p / "opencl_params.json").exists()
    )


def main():
    print(
        f"pyFAI {pyFAI.version} | rsfai.AzimuthalIntegrator (PyO3) | "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )
    print(f"numpy {np.__version__} | rel gate {REL_TOL:.0e} for accumulator fields\n")

    axis_fields = ("radial", "azimuthal")
    all_fields = axis_fields + ACCUMULATOR_FIELDS
    total_fail = 0
    total_checked = 0
    datasets_run = 0

    for d in dataset_dirs():
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        split, algo = method_of(cfg)
        if split == "pseudo":
            continue  # the 2D pseudo split is not ported
        # Only the no-split histogram is rayon-parallel; every other engine is
        # serial, so its accumulator output is bit-exact (mirrors dropin_golden.rs).
        acc_exact = (split, algo) != ("no", "histogram")

        print(f"=== {d} ===")
        rsfai_fields = run_rsfai_dropin(d, cfg)
        if rsfai_fields is None:
            print()
            continue
        live_fields = run_live(d, cfg)
        datasets_run += 1

        for field in all_fields:
            gpath = DATASETS / d / f"out_{field}.npy"
            if not gpath.exists():
                continue
            golden = np.load(gpath)
            is_axis = field in axis_fields

            r_bit, r_rel, r_nan, r_detail = compare(rsfai_fields[field], golden)
            # Axis: always bit-exact. Accumulator (rsfai leg): bit-exact for the
            # serial engines, relative <= REL_TOL for the parallel no-split
            # histogram.
            if is_axis or acc_exact:
                r_ok = r_bit
            else:
                r_ok = r_rel <= REL_TOL and r_nan == 0

            l_arr = live_fields.get(field)
            if l_arr is None:
                l_ok, l_detail = False, "live pyFAI did not expose this field"
            else:
                l_bit, _, _, l_detail = compare(l_arr, golden)
                l_ok = l_bit  # live is single-threaded: bit-exact on every field

            total_checked += 2
            total_fail += (not r_ok) + (not l_ok)
            gate = "exact" if (is_axis or acc_exact) else f"rel<={REL_TOL:.0e}"
            status = "PASS" if (r_ok and l_ok) else "FAIL"
            print(f"  out_{field:20s} {status} [{gate:>10s}] | rsfai: {r_detail} | live: {l_detail}")
        print()

    if datasets_run == 0:
        print("RESULT: FAIL — no histogram datasets the drop-in could run "
              "(check golden/datasets and the supported-detector list)")
        return 1
    print(f"ran {datasets_run} dataset(s), checked {total_checked} comparisons, {total_fail} failed")
    if total_fail:
        print("RESULT: FAIL — see the per-field detail above")
        return 1
    print("RESULT: PASS — every field of every drop-in dataset passes its gate "
          "(rsfai.AzimuthalIntegrator == pyFAI, in-process)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
