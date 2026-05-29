#!/usr/bin/env python
"""Golden generator for image reconstruction: calcfrom1d / fake_xrpdp /
fake_calibration_image.

Run single-thread in the daq env (pyFAI 2026.5.0, built -ffp-contract=off):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_calcfrom.py

Three parity surfaces feeding `crates/rsfai/tests/golden_calcfrom.rs`:

  * calcfrom1d (BIT-EXACT): reconstruct a 2D image from a 1D (2theta, intensity)
    profile via `numpy.interp` + solid-angle + flat/dark/mask. pyFAI's calcfrom1d
    is entirely f64, so given identical inputs the reconstruction is bitwise
    reproducible. Variants: solid-angle on/off, masked, flat+dark.

  * fake_xrpdp (Tier-B): a synthetic 1D powder pattern (Gaussian peaks at the
    calibrant's Bragg 2theta). The peak `exp`/`sqrt` go through numexpr, so this
    is compared at a recorded ULP/rel budget, not bitwise.

  * fake_calibration_image (Tier-B): fake_xrpdp back-projected onto the detector
    with calcfrom1d; inherits fake_xrpdp's transcendental budget.

The detector is a generic 128x128, 100 um pixels, orientation 3 (pyFAI's
default), at a short distance so several LaB6 rings fall on it. The Rust verifier
rebuilds this geometry literally (see its `build_ai`).
"""

import json
import os
import shutil

import numpy as np

import pyFAI
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.calibrant import get_calibrant
from pyFAI import units

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_calcfrom")
CALIB_DIR = os.path.join(os.path.dirname(pyFAI.__file__), "resources", "calibration")

# Geometry the Rust verifier mirrors exactly (generic detector, orientation 3).
SHAPE = (128, 128)
PIXEL = 100e-6
GEOM = dict(dist=0.02, poni1=6.4e-3, poni2=6.4e-3, rot1=0.0, rot2=0.0, rot3=0.0,
            wavelength=1e-10)


def build_ai():
    det = Detector(pixel1=PIXEL, pixel2=PIXEL, max_shape=SHAPE, orientation=3)
    return AzimuthalIntegrator(detector=det, **GEOM)


def save(meta, name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name + ".npy"), arr)
    meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    meta = {}
    ai = build_ai()

    # ---- 1. calcfrom1d (bit-exact) -------------------------------------------
    # Profile range [5, 20] deg vs detector range ~[0.2, 24.2] deg, so center
    # pixels clamp to fp[0] (left) and corners to fp[-1] (right).
    tth = np.linspace(5.0, 20.0, 40)          # deg, ascending (xp)
    intensity = 1.0 + 0.5 * np.sin(tth)       # arbitrary f64 profile (fp)
    mask = np.zeros(SHAPE, dtype=np.int8)
    mask[::7, :] = 1                          # mask every 7th row
    flat = np.full(SHAPE, 1.05, dtype=np.float64)
    dark = np.full(SHAPE, 2.0, dtype=np.float64)
    save(meta, "calcfrom1d__tth", tth)
    save(meta, "calcfrom1d__intensity", intensity)
    save(meta, "calcfrom1d__mask", mask)
    save(meta, "calcfrom1d__flat", flat)
    save(meta, "calcfrom1d__dark", dark)

    # The internal radial array calcfrom1d interpolates onto, and the result of
    # interpolating directly on it. These two isolate the parity ledger:
    #   * ttha is a transcendental geometry array (arctan2/sqrt over pixel
    #     positions) → diverges from rsFAI's by libm ULPs (Tier-B by physics).
    #   * interp_pyttha = numpy.interp(ttha, tth_internal, I) driven on THIS
    #     ttha → the Rust port reproduces it bitwise (the algebra is exact); any
    #     end-to-end divergence is therefore inherited from ttha, not the interp.
    ttha = ai.center_array(SHAPE, unit=units.TTH_DEG, scale=False).ravel()
    tth_internal = tth / units.TTH_DEG.scale
    save(meta, "calcfrom1d__ttha", ttha)
    save(meta, "calcfrom1d__interp_pyttha", np.interp(ttha, tth_internal, intensity))

    common = dict(shape=SHAPE, dim1_unit=units.TTH_DEG)
    save(meta, "calcfrom1d__img_sa",
         ai.calcfrom1d(tth, intensity, correctSolidAngle=True, **common))
    save(meta, "calcfrom1d__img_nosa",
         ai.calcfrom1d(tth, intensity, correctSolidAngle=False, **common))
    save(meta, "calcfrom1d__img_mask",
         ai.calcfrom1d(tth, intensity, correctSolidAngle=True, mask=mask, dummy=0.0, **common))
    save(meta, "calcfrom1d__img_flatdark",
         ai.calcfrom1d(tth, intensity, correctSolidAngle=True, flat=flat, dark=dark, **common))

    # ---- 2. fake_xrpdp (Tier-B) ----------------------------------------------
    shutil.copyfile(os.path.join(CALIB_DIR, "LaB6.D"), os.path.join(OUTDIR, "LaB6.D"))
    cal = get_calibrant("LaB6")
    cal.wavelength = 1e-10
    res = cal.fake_xrpdp(200, (0.0, 60.0), background=0.1, Imax=1.0,
                         resolution=0.1, unit=units.TTH_DEG)
    save(meta, "fake_xrpdp__radial", np.asarray(res.radial, dtype=np.float64))
    save(meta, "fake_xrpdp__intensity", np.asarray(res.intensity, dtype=np.float64))

    # ---- 3. fake_calibration_image (Tier-B) ----------------------------------
    cal2 = get_calibrant("LaB6")
    cal2.wavelength = 1e-10
    img = cal2.fake_calibration_image(ai, shape=SHAPE, Imax=1.0, Imin=0.1, resolution=0.1)
    save(meta, "fake_cal_image__img", np.asarray(img, dtype=np.float64))

    manifest = {
        "pyfai_version": pyFAI.version,
        "numpy_version": np.__version__,
        "shape": list(SHAPE),
        "pixel": PIXEL,
        "geometry": GEOM,
        "fake_xrpdp": {"nbpt": 200, "tth_range_deg": [0.0, 60.0],
                       "background": 0.1, "imax": 1.0, "resolution_deg": 0.1,
                       "calibrant": "LaB6", "n_2th": int(len(cal.get_2th()))},
        "fake_cal_image": {"imax": 1.0, "imin": 0.1, "resolution_deg": 0.1},
        "arrays": meta,
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote calcfrom golden to {OUTDIR}")
    print(f"  calcfrom1d: 4 image variants ({SHAPE[0]}x{SHAPE[1]})")
    print(f"  fake_xrpdp: 200-pt LaB6 pattern, {len(cal.get_2th())} rings")
    print(f"  fake_calibration_image: {SHAPE[0]}x{SHAPE[1]}, "
          f"max={float(np.max(img)):.4f} min={float(np.min(img)):.4f}")


if __name__ == "__main__":
    main()
