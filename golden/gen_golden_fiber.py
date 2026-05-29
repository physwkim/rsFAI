#!/usr/bin/env python
"""Golden generator for the fiber / grazing-incidence unit equations (M10.0).

Run single-thread in the daq env (scipy 1.17.1 present):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_fiber.py

This gates `rsfai_geometry::fiber_units` as a PURE equation test (identical-input
Tier A, ULP-budgeted): it builds a real Pilatus1M geometry, then dumps a strided
sample of ~4096 pixels' lab coords `(x, y, z)` plus pyFAI's per-unit center value
for each `(incident_angle, tilt_angle, sample_orientation)` combo. The Rust
verifier feeds the SAME `(x, y, z)` to `fiber_equation` and compares with a
measured ULP budget — pyFAI evaluates `unit.equation` (scipy `Rotation`, a
quaternion), rsFAI uses a direct cos/sin matrix, so the two diverge by a small,
measured ULP amount (NOT bit-exact); see the crate docs / kodex f3389aef.

Dumps (committed): sample_{x,y,z}.npy + out_<combo>__<key>.npy (each ~4096 f64,
~32 KB) + manifest.json. No large per-pixel detector arrays are committed.
"""

import json
import os

import numpy as np

from pyFAI import units as U
from pyFAI.detectors import detector_factory
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_fiber")

# Unscaled pyFAI fiber-unit equations, keyed by the rsFAI FiberSpace name.
EQS = {
    "qip": U.eq_qip,
    "qoop": U.eq_qoop,
    "qtot": U.eq_q_total,
    "chigi": U.eq_chi_gi,
    "qbeam": U.eq_qbeam,
    "qhorz": U.eq_qhorz,
    "scatvert": U.eq_scattering_angle_vertical,
    "scathorz": U.eq_scattering_angle_horz,
    "exitvert": U.eq_exit_angle_vert,
    "exithorz": U.eq_exit_angle_horz,
}

# (incident_angle, tilt_angle, sample_orientation): no-GI baseline, GI on the
# identity orientation, three orientation remaps incl. a negative tilt.
COMBOS = [
    (0.0, 0.0, 1),
    (0.2, 0.4, 1),
    (0.2, 0.4, 2),
    (0.2, 0.4, 5),
    (0.1, -0.3, 7),
]

N_SAMPLE = 4096


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    det = detector_factory("Pilatus1M")
    ai = AzimuthalIntegrator(
        dist=0.2,
        poni1=0.1,
        poni2=0.1,
        rot1=0.01,
        rot2=0.02,
        rot3=0.0,
        detector=det,
        wavelength=1e-10,
    )
    wl = ai.wavelength

    # Lab coords for every pixel centre (pos = (z, y, x), matching center_array).
    pos = ai.position_array(corners=False)
    x_full = pos[..., 2].ravel()
    y_full = pos[..., 1].ravel()
    z_full = pos[..., 0].ravel()
    npix = x_full.size

    # A deterministic strided sample spanning the whole detector.
    stride = max(1, npix // N_SAMPLE)
    idx = np.arange(0, npix, stride)[:N_SAMPLE]
    x = np.ascontiguousarray(x_full[idx], dtype=np.float64)
    y = np.ascontiguousarray(y_full[idx], dtype=np.float64)
    z = np.ascontiguousarray(z_full[idx], dtype=np.float64)

    np.save(os.path.join(OUTDIR, "sample_x.npy"), x)
    np.save(os.path.join(OUTDIR, "sample_y.npy"), y)
    np.save(os.path.join(OUTDIR, "sample_z.npy"), z)

    combos_meta = []
    for ci, (inc, tilt, so) in enumerate(COMBOS):
        for key, fn in EQS.items():
            ary = fn(
                x=x,
                y=y,
                z=z,
                wavelength=wl,
                incident_angle=inc,
                tilt_angle=tilt,
                sample_orientation=so,
            )
            ary = np.ascontiguousarray(np.atleast_1d(ary), dtype=np.float64)
            np.save(os.path.join(OUTDIR, f"out_c{ci}__{key}.npy"), ary)
        combos_meta.append(
            {"index": ci, "incident_angle": inc, "tilt_angle": tilt, "sample_orientation": so}
        )

    manifest = {
        "dataset": "fiber_units",
        "source": "gen_golden_fiber.py",
        "wavelength": wl,
        "n_sample": int(idx.size),
        "units": list(EQS.keys()),
        "combos": combos_meta,
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(COMBOS)}x{len(EQS)} fiber golden arrays ({idx.size} pixels) to {OUTDIR}")


if __name__ == "__main__":
    main()
