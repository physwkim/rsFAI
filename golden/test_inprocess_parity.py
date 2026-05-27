#!/usr/bin/env python3
"""In-process side-by-side parity: the rsfai PyO3 kernels vs pyFAI, bit-for-bit.

Run in the ``daq`` conda env (which has both ``pyFAI`` and the maturin-built
``rsfai`` installed):

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/test_inprocess_parity.py

For each committed golden dataset, in ONE interpreter importing both libraries:

  * **rsfai** consumes the dumped Tier-A intermediates — the per-pixel arrays
    pyFAI's geometry + preproc produced (``pos0_center_unscaled``, ``chi_center``,
    ``corners``, ``preproc``, ``mask``). For the CSR paths it builds the LUT
    (and the built ``data``/``indices``/``indptr`` is checked against the
    committed ``csr_*``) then applies it; for the histogram paths it runs the
    binning + reduction directly.
  * **pyFAI (live)** re-runs ``ai.integrate1d_ng`` / ``integrate2d_ng`` on the
    dumped image in this same process.

Every exposed output field is asserted bit-identical across BOTH
``rsfai == committed golden`` AND ``live pyFAI == committed golden`` (so, by
transitivity, ``rsfai == live pyFAI``). Bit-for-bit is the target; any
divergence is reported as a max-ULP figure and the test fails — tolerance is
never silently widened. The live-vs-golden leg is the libm anchor: if it
diverges, the env's transcendentals drifted from the golden's, and the ULP
figure quantifies it.

Exit code is 0 only if every field of every dataset is bit-exact.
"""

import json
import math
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

# Golden output field -> the rsfai unified-dict key holding the same quantity.
# (radial/azimuthal are handled separately because they carry the unit scale.)
FIELD_KEYS_1D = {
    "intensity": "intensity",
    "sigma": "sigma",
    "count": "count",
    "sum_signal": "sum_signal",
    "sum_variance": "sum_variance",
    "sum_normalization": "sum_normalization",
    "sum_normalization2": "sum_normalization2",
    "std": "std",
    "sem": "sem",
}
FIELD_KEYS_2D = dict(FIELD_KEYS_1D)  # same field names; values are 2D arrays


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


def compare(actual, golden):
    """Return (ok, detail). ok iff byte-identical; detail carries the ULP report."""
    a = np.ascontiguousarray(actual)
    g = np.ascontiguousarray(golden)
    if a.dtype != g.dtype:
        return False, f"dtype {a.dtype} != golden {g.dtype}"
    if a.shape != g.shape:
        return False, f"shape {a.shape} != golden {g.shape}"
    if a.tobytes() == g.tobytes():
        return True, "bit-exact"
    # Byte-divergent: quantify in ULPs over the mismatching elements.
    af = a.ravel()
    gf = g.ravel()
    is_f32 = a.dtype == np.float32
    if a.dtype not in (np.float32, np.float64):
        n = int(np.count_nonzero(af != gf))
        return False, f"{n} of {af.size} elements differ (non-float dtype {a.dtype})"
    diff_idx = np.nonzero(af != gf)[0]
    max_ulp = 0
    n_nan = 0
    for i in diff_idx:
        xv, yv = af[i], gf[i]
        if np.isnan(xv) or np.isnan(yv):
            n_nan += 1
            continue
        u = _ulp_scalar(xv, yv, is_f32)
        if u > max_ulp:
            max_ulp = u
    nan_note = f", {n_nan} NaN-bit" if n_nan else ""
    return False, f"{diff_idx.size} of {af.size} differ, max_ulp={max_ulp}{nan_note}"


def method_of(cfg):
    m = tuple(cfg["method"])
    return m[0], m[1]  # (split, algo)


def run_rsfai(d, cfg):
    """Run the rsfai kernel chain for one dataset; return a unified field dict.

    For CSR datasets the second return value is (built_data, built_indices,
    built_indptr) so the caller can check the build against the committed LUT;
    histogram datasets return None there.
    """
    dim = cfg.get("dim", 1)
    split, algo = method_of(cfg)
    em = cfg["error_model_code"]
    unit_scale = cfg["unit_scale"]
    mask = np.ascontiguousarray(load(d, "mask").reshape(-1))
    prep = np.ascontiguousarray(load(d, "preproc").reshape(-1, 4))

    if algo == "histogram":
        if dim == 1:
            radial = np.ascontiguousarray(load(d, "pos0_center_unscaled").reshape(-1))
            if split == "bbox":
                # Direct-split bbox histogram: f64 binned sums (CsrIntegrate1d keys).
                dpos0 = np.ascontiguousarray(load(d, "pos0_delta").reshape(-1))
                out = rsfai.histogram1d_bbox(
                    radial, dpos0, prep, mask=mask, npt=cfg["npt"],
                    error_model=em, empty=0.0, allow_pos0_neg=False,
                )
                fields = {
                    "radial": out["position"] * unit_scale,
                    "intensity": out["intensity"],
                    "sigma": out["sigma"],
                    "count": out["count"],
                    "sum_signal": out["sum_signal"],
                    "sum_variance": out["sum_variance"],
                    "sum_normalization": out["sum_normalization"],
                    "sum_normalization2": out["sum_norm_sq"],
                    "std": out["std"],
                    "sem": out["sem"],
                }
                return fields, None
            if split == "full":
                # Full pixel-splitting histogram: corners widened to f64, flattened.
                # 1D setup does not forward chiDiscAtPi/pos1_period (constructor
                # defaults: chiDiscAtPi=True, pos1_period=2π).
                corners = np.ascontiguousarray(load(d, "corners").astype(np.float64).reshape(-1))
                out = rsfai.histogram1d_full(
                    corners, prep, mask=mask, npt=cfg["npt"],
                    error_model=em, empty=0.0, allow_pos0_neg=False,
                    chi_disc_at_pi=True, pos1_period=2.0 * math.pi,
                )
                fields = {
                    "radial": out["position"] * unit_scale,
                    "intensity": out["intensity"],
                    "sigma": out["sigma"],
                    "count": out["count"],
                    "sum_signal": out["sum_signal"],
                    "sum_variance": out["sum_variance"],
                    "sum_normalization": out["sum_normalization"],
                    "sum_normalization2": out["sum_norm_sq"],
                    "std": out["std"],
                    "sem": out["sem"],
                }
                return fields, None
            out = rsfai.histogram1d(radial, prep, cfg["npt"], error_model=em, empty=0.0)
            fields = {
                "radial": out["position"] * unit_scale,
                "intensity": out["intensity"],
                "sigma": out["sigma"],
                "count": out["count"],
                "sum_signal": out["signal"],
                "sum_variance": out["variance"],
                "sum_normalization": out["normalization"],
                "sum_normalization2": out["norm_sq"],
                "std": out["std"],
                "sem": out["sem"],
            }
            return fields, None
        # 2D histogram
        if split == "bbox":
            return run_rsfai_2d_bbox_histogram(d, cfg, prep, mask), None
        if split == "full":
            return run_rsfai_2d_full_histogram(d, cfg, prep, mask), None
        return run_rsfai_2d_histogram(d, cfg, prep, mask), None

    # CSR build + apply
    npt = (cfg["npt_rad"], cfg["npt_azim"]) if dim == 2 else cfg["npt"]
    if split in ("no", "bbox"):
        # pyFAI's ("no","csr",…) and ("bbox","csr",…) share the same HistoBBox
        # class; no-split passes delta=None (do_split=False), collapsing each
        # pixel to one coef-1.0 entry. Mirror that here: same builder, deltas
        # only for the bbox split.
        do_split = split == "bbox"
        if dim == 1:
            pos0 = np.ascontiguousarray(load(d, "pos0_center_unscaled").reshape(-1))
            dpos0 = np.ascontiguousarray(load(d, "pos0_delta").reshape(-1)) if do_split else None
            data, indices, indptr, bc = rsfai.build_bbox_csr_1d(
                pos0, delta_pos0=dpos0, mask=mask, bins=npt, allow_pos0_neg=False
            )
        else:
            pos0 = np.ascontiguousarray(load(d, "pos0_center_unscaled").reshape(-1))
            pos1 = np.ascontiguousarray(load(d, "chi_center").reshape(-1))
            dpos0 = np.ascontiguousarray(load(d, "pos0_delta").reshape(-1)) if do_split else None
            dpos1 = np.ascontiguousarray(load(d, "chi_delta").reshape(-1)) if do_split else None
            data, indices, indptr, bc0, bc1 = rsfai.build_bbox_csr_2d(
                pos0, pos1, delta_pos0=dpos0, delta_pos1=dpos1, mask=mask, bins=npt,
                allow_pos0_neg=False, chi_disc_at_pi=cfg["chi_disc_at_pi"],
                pos1_period=cfg["pos1_period"],
            )
    else:  # full split: corners (f32) widened to f64, flattened
        corners = np.ascontiguousarray(load(d, "corners").astype(np.float64).reshape(-1))
        if dim == 1:
            data, indices, indptr, bc = rsfai.build_full_csr_1d(
                corners, mask=mask, bins=npt, allow_pos0_neg=False,
                chi_disc_at_pi=True, pos1_period=2.0 * math.pi,
            )
        else:
            data, indices, indptr, bc0, bc1 = rsfai.build_full_csr_2d(
                corners, mask=mask, bins=npt, allow_pos0_neg=False,
                chi_disc_at_pi=cfg["chi_disc_at_pi"], pos1_period=cfg["pos1_period"],
            )

    built = (data, indices, indptr)
    if dim == 1:
        out = rsfai.csr_integrate1d(data, indices, indptr, prep, bc, error_model=em, empty=0.0)
        fields = {
            "radial": out["position"] * unit_scale,
            "intensity": out["intensity"],
            "sigma": out["sigma"],
            "count": out["count"],
            "sum_signal": out["sum_signal"],
            "sum_variance": out["sum_variance"],
            "sum_normalization": out["sum_normalization"],
            "sum_normalization2": out["sum_norm_sq"],
            "std": out["std"],
            "sem": out["sem"],
        }
        return fields, built
    out = rsfai.csr_integrate2d(data, indices, indptr, prep, bc0, bc1, error_model=em, empty=0.0)
    return _fields_2d(out, cfg), built


def run_rsfai_2d_histogram(d, cfg, prep, mask):
    radial = np.ascontiguousarray(load(d, "pos0_center_unscaled").reshape(-1))
    azimuthal = np.ascontiguousarray(load(d, "chi_center").reshape(-1))
    out = rsfai.histogram2d(
        radial, azimuthal, prep, bins=(cfg["npt_rad"], cfg["npt_azim"]), mask=mask,
        error_model=cfg["error_model_code"], allow_radial_neg=False,
        chi_disc_at_pi=cfg["chi_disc_at_pi"], pos1_period=cfg["pos1_period"], empty=0.0,
    )
    return _fields_2d(out, cfg)


def run_rsfai_2d_bbox_histogram(d, cfg, prep, mask):
    radial = np.ascontiguousarray(load(d, "pos0_center_unscaled").reshape(-1))
    dpos0 = np.ascontiguousarray(load(d, "pos0_delta").reshape(-1))
    azimuthal = np.ascontiguousarray(load(d, "chi_center").reshape(-1))
    dpos1 = np.ascontiguousarray(load(d, "chi_delta").reshape(-1))
    out = rsfai.histogram2d_bbox(
        radial, dpos0, azimuthal, dpos1, prep,
        bins=(cfg["npt_rad"], cfg["npt_azim"]), mask=mask, allow_pos0_neg=False,
        chi_disc_at_pi=cfg["chi_disc_at_pi"], pos1_period=cfg["pos1_period"],
        error_model=cfg["error_model_code"], empty=0.0,
    )
    return _fields_2d(out, cfg)


def run_rsfai_2d_full_histogram(d, cfg, prep, mask):
    # Full pixel-splitting histogram: corners widened to f64, flattened. 2D setup
    # forwards chiDiscAtPi and pos1_period = unit1.period (360, applied to radian
    # azimuths — a pyFAI quirk).
    corners = np.ascontiguousarray(load(d, "corners").astype(np.float64).reshape(-1))
    out = rsfai.histogram2d_full(
        corners, prep, bins=(cfg["npt_rad"], cfg["npt_azim"]), mask=mask,
        allow_pos0_neg=False, chi_disc_at_pi=cfg["chi_disc_at_pi"],
        pos1_period=cfg["pos1_period"], error_model=cfg["error_model_code"], empty=0.0,
    )
    return _fields_2d(out, cfg)


def _fields_2d(out, cfg):
    """Map an rsfai 2D output dict to the golden field layout (npt_azim, npt_rad)."""
    npt_rad, npt_azim = cfg["npt_rad"], cfg["npt_azim"]
    unit_scale, azim_scale = cfg["unit_scale"], cfg["azim_scale"]

    def grid(flat):
        return np.asarray(flat).reshape(npt_azim, npt_rad)

    return {
        "radial": np.asarray(out["radial"]) * unit_scale,
        "azimuthal": np.asarray(out["azimuthal"]) * azim_scale,
        "intensity": grid(out["intensity"]),
        "sigma": grid(out["sigma"]),
        "count": grid(out["count"]),
        "sum_signal": grid(out["signal"]),
        "sum_variance": grid(out["variance"]),
        "sum_normalization": grid(out["normalization"]),
        "sum_normalization2": grid(out["norm_sq"]),
        "std": grid(out["std"]),
        "sem": grid(out["sem"]),
    }


def run_live(d, cfg):
    """Re-run pyFAI's high-level integrator in-process; return field-name -> array."""
    ai = pyFAI.load(str(DATASETS / d / "geometry.poni"))
    img = load(d, "image")
    common = dict(
        unit=cfg["unit"], method=tuple(cfg["method"]),
        correctSolidAngle=cfg["correct_solid_angle"], error_model=cfg["error_model"],
        polarization_factor=cfg["polarization_factor"],
        normalization_factor=cfg["normalization_factor"],
        radial_range=cfg["radial_range"], azimuth_range=cfg["azimuth_range"],
    )
    if cfg.get("dim", 1) == 2:
        res = ai.integrate2d_ng(img, cfg["npt_rad"], cfg["npt_azim"], **common)
    else:
        res = ai.integrate1d_ng(img, cfg["npt"], **common)
    attrs = ("radial", "azimuthal", "intensity", "sigma", "count", "sum_signal",
             "sum_variance", "sum_normalization", "sum_normalization2", "std", "sem")
    out = {}
    for a in attrs:
        v = getattr(res, a, None)
        if isinstance(v, np.ndarray):
            out[a] = v
    return out


def dataset_dirs():
    return sorted(p.name for p in DATASETS.iterdir() if (p / "manifest.json").exists())


def main():
    print(f"pyFAI {pyFAI.version} | rsfai (PyO3) | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"numpy {np.__version__}\n")

    all_fields = ["radial", "azimuthal"] + list(FIELD_KEYS_1D.keys())
    total_fail = 0
    total_checked = 0

    for d in dataset_dirs():
        cfg = json.load(open(DATASETS / d / "manifest.json"))["config"]
        print(f"=== {d} ===")

        rsfai_fields, built = run_rsfai(d, cfg)
        live_fields = run_live(d, cfg)

        # CSR build: the rsfai-built LUT vs the committed pyFAI LUT.
        if built is not None:
            data, indices, indptr = built
            for nm, act in (("csr_data", data), ("csr_indices", indices), ("csr_indptr", indptr)):
                ok, detail = compare(act, load(d, nm))
                total_checked += 1
                total_fail += not ok
                print(f"  build {nm:14s} rsfai==golden : {'PASS' if ok else 'FAIL'} ({detail})")

        # Output fields: rsfai==golden AND live pyFAI==golden (=> rsfai==live).
        for field in all_fields:
            gpath = DATASETS / d / f"out_{field}.npy"
            if not gpath.exists():
                continue
            golden = np.load(gpath)

            r_ok, r_detail = compare(rsfai_fields[field], golden)
            l_arr = live_fields.get(field)
            if l_arr is None:
                l_ok, l_detail = False, "live pyFAI did not expose this field"
            else:
                l_ok, l_detail = compare(l_arr, golden)

            total_checked += 2
            total_fail += (not r_ok) + (not l_ok)
            status = "PASS" if (r_ok and l_ok) else "FAIL"
            print(f"  out_{field:20s} {status}  | rsfai: {r_detail} | live: {l_detail}")
        print()

    print(f"checked {total_checked} comparisons, {total_fail} failed")
    if total_fail:
        print("RESULT: FAIL — see the per-field detail above (bit-exact is the gate)")
        return 1
    print("RESULT: PASS — every field of every dataset is bit-identical "
          "(rsfai == pyFAI, in-process)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
