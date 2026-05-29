#!/usr/bin/env python
"""Golden generator for FiberIntegrator.integrate2d_fiber / integrate_fiber (M10.1).

Run single-thread in the daq env (numexpr present):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_fiber_integrator.py

Builds a real Pilatus1M FiberIntegrator, a deterministic detector image, and
dumps the 2D fiber map + the two 1D folds (vertical / horizontal) for a matrix
of grazing-incidence params and fiber-unit pairs. The method is the fiber
default `("no","histogram","cython")` (no pixel splitting). pyFAI's fiber path
does not plumb an error model, so `sum_variance`/`std`/`sem` are `None`; the
manifest records, per result, which fields came back `None` so the Rust verifier
gates only the populated ones.

The fiber position arrays (qip/qoop) carry the scipy-quaternion-vs-direct-matrix
ULP divergence (see datasets_fiber / kodex f3389aef); whether that propagates
through the histogram bin assignment is what the verifier measures.
"""

import json
import os

import numpy as np

from pyFAI.detectors import detector_factory
from pyFAI.integrator.fiber import FiberIntegrator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_fiber_integrator")

GEOM = dict(dist=0.2, poni1=0.1, poni2=0.1, rot1=0.01, rot2=0.02, rot3=0.0, wavelength=1e-10)
METHOD = ("no", "histogram", "cython")

# (tag, incident_angle, tilt_angle, sample_orientation, unit_ip, unit_oop,
#  npt_ip, npt_oop, correctSolidAngle)
COMBOS = [
    ("qnm_so1_sa", 0.2, 0.0, 1, "qip_nm^-1", "qoop_nm^-1", 200, 200, True),
    ("qnm_tilt_so1", 0.2, 0.1, 1, "qip_nm^-1", "qoop_nm^-1", 150, 180, True),
    ("qA_so2", 0.15, -0.05, 2, "qip_A^-1", "qoop_A^-1", 200, 200, True),
    ("exit_deg_so1", 0.2, 0.0, 1, "exit_angle_horz_deg", "exit_angle_vert_deg", 120, 120, True),
    ("qnm_nosa", 0.2, 0.1, 1, "qip_nm^-1", "qoop_nm^-1", 200, 200, False),
]


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name), arr)
    return list(arr.shape), str(arr.dtype)


def dump_attrs(prefix, res, names):
    """Save each non-None attribute of `res`; return {name: shape|null}."""
    out = {}
    for n in names:
        v = getattr(res, n, None)
        if v is None:
            out[n] = None
        else:
            out[n] = save(f"{prefix}__{n}.npy", np.asarray(v))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    det = detector_factory("Pilatus1M")
    fi = FiberIntegrator(detector=det, **GEOM)

    # A .poni so the Rust verifier rebuilds the identical geometry + detector
    # via AzimuthalIntegrator::load (the proven detector-resolution path).
    fi.save(os.path.join(OUTDIR, "geometry.poni"))

    rng = np.random.default_rng(12345)
    data = (rng.random(det.shape) * 1000.0).astype(np.float32)
    save("data.npy", data)

    combos_meta = []
    for (tag, inc, tilt, so, unit_ip, unit_oop, npt_ip, npt_oop, csa) in COMBOS:
        gi = dict(
            sample_orientation=so,
            incident_angle=inc,
            tilt_angle=tilt,
            angle_unit="rad",
        )
        res2d = fi.integrate2d_fiber(
            data, npt_ip=npt_ip, unit_ip=unit_ip, npt_oop=npt_oop, unit_oop=unit_oop,
            correctSolidAngle=csa, method=METHOD, **gi,
        )
        f2d = {}
        f2d["intensity"] = save(f"{tag}__2d__intensity.npy", np.asarray(res2d.intensity))
        f2d["inplane"] = save(f"{tag}__2d__inplane.npy", np.asarray(res2d.inplane))
        f2d["outofplane"] = save(f"{tag}__2d__outofplane.npy", np.asarray(res2d.outofplane))
        f2d.update(
            dump_attrs(
                f"{tag}__2d",
                res2d,
                ["sum_signal", "sum_normalization", "sum_normalization2",
                 "sum_variance", "count", "std", "sem"],
            )
        )

        ones = {}
        for vert in (True, False):
            vtag = "v" if vert else "h"
            res1d = fi.integrate_fiber(
                data, npt_ip=npt_ip, unit_ip=unit_ip, npt_oop=npt_oop, unit_oop=unit_oop,
                vertical_integration=vert, correctSolidAngle=csa, method=METHOD, **gi,
            )
            f1 = {}
            f1["integrated"] = save(f"{tag}__1d{vtag}__integrated.npy", np.asarray(res1d.integrated))
            f1["intensity"] = save(f"{tag}__1d{vtag}__intensity.npy", np.asarray(res1d.intensity))
            sig = res1d.sigma
            f1["sigma"] = None if sig is None else save(f"{tag}__1d{vtag}__sigma.npy", np.asarray(sig))
            f1.update(
                dump_attrs(
                    f"{tag}__1d{vtag}",
                    res1d,
                    ["sum_signal", "sum_normalization", "count", "sum_variance"],
                )
            )
            ones[vtag] = f1

        combos_meta.append(
            {
                "tag": tag,
                "incident_angle": inc,
                "tilt_angle": tilt,
                "sample_orientation": so,
                "unit_ip": unit_ip,
                "unit_oop": unit_oop,
                "npt_ip": npt_ip,
                "npt_oop": npt_oop,
                "correct_solid_angle": csa,
                "result2d": f2d,
                "result1d": ones,
            }
        )

    manifest = {
        "dataset": "fiber_integrator",
        "source": "gen_golden_fiber_integrator.py",
        "detector": "Pilatus1M",
        "geometry": GEOM,
        "method": list(METHOD),
        "shape": list(det.shape),
        "combos": combos_meta,
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(COMBOS)} fiber-integrator combos to {OUTDIR}")
    # Report which result fields came back None (drives the Rust struct shape).
    for c in combos_meta:
        nulls2d = [k for k, v in c["result2d"].items() if v is None]
        print(f"  {c['tag']}: 2D None fields = {nulls2d or 'none'}")


if __name__ == "__main__":
    main()
