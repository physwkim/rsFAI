#!/usr/bin/env python
"""Golden generator for Dioptas's automatic refinement (rsfai-calib ring_search).

Run headless, single-thread, with Dioptas 0.13.0 on pyFAI 2026.9.0, the
release whose Bilinear.local_maxi Hessian rsfai-peaks follows (the
micromamba env pyfai2026.9 holds both as pip wheels):

    QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 \\
        ~/micromamba/envs/pyfai2026.9/bin/python -I golden/gen_golden_ringsearch.py

Dioptas does the work. A synthetic LaB6 image is loaded into its
CalibrationModel, which runs search_peaks_on_ring and refine in the order of
CalibrationController.automatic_refinement (the controller is bound to its
widgets, so that loop is restated below). numpy.random.shuffle and
Massif.peaks_from_area are wrapped to record, for each ring: the geometry
it was searched with, the band (mask2, Imin, keep), the visit order and
the peaks kept.

Surfaces, all feeding crates/rsfai-calib/tests/golden_ringsearch.rs:

  * tth: Geometry.twoThetaArray at the start geometry (ULP tolerance).
  * band: per ring, mask2 from the geometry the ring was searched with
    (exact but at the band edge, where pyFAI's float32 pixel positions
    decide), Imin (tolerance: numpy sums pairwise), keep (exact).
  * massif: per ring, the peaks Massif keeps visiting the recorded order
    (bit-exact).
  * refine: the geometry after the last ring, against Dioptas's and the
    truth the image was drawn from (tolerance: SLSQP vs Nelder-Mead).

The image is 128x128, so the whole tree is committed.
"""

import json
import os
import platform
import tempfile

# Before numpy loads: its AVX512 kernels (SVML pow/arctan2/arccos) are not
# libm-exact, so a golden would depend on the generating CPU.
os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"

import fabio
import numpy as np
import scipy

import pyFAI
from pyFAI.calibrant import Calibrant
from pyFAI.detectors import Detector
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.massif import Massif

import dioptas
from dioptas.model.CalibrationModel import CalibrationModel, NotEnoughSpacingsInCalibrant
from dioptas.model.ImgModel import ImgModel
from dioptas.paths import calibrants_path

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_ringsearch")

SHAPE = (128, 128)
PIXEL = 1.5e-4
WAVELENGTH = 0.25e-10
# The geometry the image is drawn from, and the start Dioptas refines from:
# off by a fraction of a pixel, inside the 0.1 deg band.
TRUTH = dict(dist=0.1, poni1=0.0098, poni2=0.0094, rot1=0.012, rot2=-0.009, rot3=0.0)
START = dict(dist=0.1004, poni1=0.0099, poni2=0.00932, rot1=0.0, rot2=0.0, rot3=0.0)
NUM_RINGS = 6
DELTA_TTH = 0.1
MEAN_FACTOR = 1.0
UPPER = 55000.0
SEED = 20261109

KEYS = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")


def save(name, arr):
    np.save(os.path.join(OUTDIR, name), arr)


def param7(geo):
    return np.array([getattr(geo, k) for k in KEYS] + [geo.wavelength], dtype=np.float64)


def synth():
    """LaB6 rings with a six-fold texture on a flat background, plus noise."""
    det = Detector(PIXEL, PIXEL, max_shape=SHAPE, orientation=3)
    ai = AzimuthalIntegrator(detector=det, wavelength=WAVELENGTH, **TRUTH)
    cal = Calibrant(filename=os.path.join(calibrants_path, "LaB6.D"))
    cal.wavelength = WAVELENGTH
    rings = cal.fake_calibration_image(ai, shape=SHAPE, Imax=1000.0, Imin=0.0, resolution=0.05)
    chi = ai.center_array(SHAPE, unit="chi_rad", scale=False)
    rng = np.random.default_rng(SEED)
    img = 40.0 + rings * (0.55 + 0.45 * np.cos(3.0 * chi) ** 2) + rng.normal(0.0, 4.0, SHAPE)
    return img.astype(np.float32)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    image = synth()
    save("image.npy", image)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lab6.edf")
        # ImgModel.load flips the file upside down; store it flipped so the
        # image Dioptas searches is `image`.
        fabio.edfimage.EdfImage(data=np.flipud(image)).write(path)
        img_model = ImgModel()
        img_model.load(path)
    assert img_model.img_data.dtype == np.float32
    assert np.array_equal(img_model.img_data, image)

    cm = CalibrationModel(img_model)
    cm.set_pixel_size((PIXEL, PIXEL))
    cm.update_detector_shape()
    cm.set_calibrant(os.path.join(calibrants_path, "LaB6.D"))
    cm.set_fixed_values({})
    cm.pattern_geometry = GeometryRefinement(
        detector=cm.detector, calibrant=cm.calibrant, wavelength=WAVELENGTH, **START
    )
    cm.is_calibrated = True
    # load_calibrant(wavelength_from="pyFAI")
    cm.calibrant.setWavelength_change2th(cm.pattern_geometry.wavelength)
    save("calibrant_dspacing.npy", np.array(cm.calibrant.dspacing, dtype=np.float64))
    save("calibrant_2th.npy", np.array(cm.calibrant.get_2th(), dtype=np.float64))
    save("start_param.npy", param7(cm.pattern_geometry))
    save("truth_param.npy", np.array([TRUTH[k] for k in KEYS] + [WAVELENGTH]))
    save("tth_start.npy", np.asarray(cm.pattern_geometry.twoThetaArray(SHAPE), dtype=np.float64))

    record = {}
    shuffle = np.random.shuffle

    def recording_shuffle(a):
        shuffle(a)
        record["order"] = np.array(a, copy=True)

    peaks_from_area = Massif.peaks_from_area

    def recording_peaks(self, mask, Imin, keep, **kw):
        record["order"] = np.zeros((0, 2), dtype=np.int64)
        res = peaks_from_area(self, mask, Imin=Imin, keep=keep, **kw)
        record.update(mask=np.array(mask, copy=True), imin=float(Imin), keep=int(keep), peaks=list(res))
        return res

    np.random.shuffle = recording_shuffle
    Massif.peaks_from_area = recording_peaks
    np.random.seed(SEED)

    cm.setup_peak_search_algorithm("Massif")
    rings = []

    def search(ring):
        record.clear()
        before = param7(cm.pattern_geometry)
        cm.search_peaks_on_ring(ring, DELTA_TTH, MEAN_FACTOR, UPPER, None)
        if "mask" not in record:
            return
        w = SHAPE[1]
        order = record["order"]
        save(f"ring{ring}_param.npy", before)
        save(f"ring{ring}_area.npy", np.flatnonzero(record["mask"]).astype(np.int32))
        save(f"ring{ring}_order.npy", (order[:, 0] * w + order[:, 1]).astype(np.int32))
        save(f"ring{ring}_peaks.npy", np.array(record["peaks"], dtype=np.float64).reshape(-1, 2))
        save(f"ring{ring}_cut.npy", np.array([record["imin"], record["keep"]], dtype=np.float64))
        rings.append(dict(ring=ring, imin=record["imin"], keep=record["keep"], peaks=len(record["peaks"])))

    # CalibrationController.automatic_refinement
    search(0)
    search(1)
    if len(cm.points):
        cm.refine()
    for i in range(NUM_RINGS - 2):
        try:
            search(i + 2)
        except NotEnoughSpacingsInCalibrant:
            break
        if len(cm.points):
            cm.refine()

    np.random.shuffle = shuffle
    Massif.peaks_from_area = peaks_from_area

    final = param7(cm.pattern_geometry)
    points = cm.get_point_array().astype(np.float64)
    save("final_param.npy", final)
    save("final_points.npy", points)
    manifest = dict(
        dataset="ringsearch_LaB6_synthetic",
        dioptas_version=dioptas.__version__,
        pyfai_version=pyFAI.version,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        platform=platform.platform(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        config=dict(
            shape=list(SHAPE),
            pixel=PIXEL,
            wavelength=WAVELENGTH,
            orientation=3,
            truth=TRUTH,
            start=START,
            num_rings=NUM_RINGS,
            delta_tth_deg=DELTA_TTH,
            min_mean_factor=MEAN_FACTOR,
            upper_limit=UPPER,
            seed=SEED,
        ),
        rings=rings,
        final_param=final.tolist(),
        final_chi2=float(cm.pattern_geometry.chi2()),
        npt=int(len(points)),
    )
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(dict(rings=rings, final=final.tolist(), chi2=manifest["final_chi2"]), indent=1))


if __name__ == "__main__":
    main()
