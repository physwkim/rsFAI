#!/usr/bin/env python
"""Golden generator for Calibrant + crystallography (M12).

Run single-thread in the daq env (pyFAI 2026.5.0, built -ffp-contract=off):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_calibrant.py

Two parity surfaces, both feeding the Rust verifier in
`crates/rsfai-calibrant/tests/golden_calibrant.rs`:

  * `.D`-file path: for a handful of shipped calibrants (AgBh, Al, LaB6, Si,
    CeO2) we copy the `.D` file into `datasets_calibrant/`, then for a matrix of
    wavelengths dump the d-spacing list, the Bragg 2theta list (`get_2th`,
    radians), and `get_peaks` in each supported unit. The Rust side re-parses the
    *same* committed `.D` file and recomputes, so the d-spacing parse and the
    `2*asin(5e9*lambda/d)` ring positions are compared directly.

  * Cell path: build `Cell.cubic` / `Cell.diamond` for the cubic calibrants and
    dump `calculate_dspacing` (the d-spacing list down to dmin) so the Rust
    `Cell` reproduces the same lattice -> d-spacing arithmetic that generated the
    shipped `.D` files. Also covers the two R-centered space-group conditions the
    tutorials append to a primitive hexagonal cell: `group167_R3bar_c` (R-3c, the
    corundum-type Cr2O3 / eskolaite) and `group166_R3bar_m` (R-3m,
    hydrocerussite). The Rust side mirrors them via `Cell::add_selection_rule`.

Provenance (pyFAI/numpy version, CONST_hc, per-field dtype) goes in
`manifest.json`. CONST_hc is dumped explicitly so a scipy.constants change fails
the Rust verifier loudly rather than drifting silently.
"""

import json
import os
import shutil

import numpy as np

import pyFAI
from pyFAI.calibrant import get_calibrant
from pyFAI.crystallography.cell import Cell
from pyFAI.crystallography.space_groups import ReflectionCondition
from pyFAI.units import CONST_hc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_calibrant")
CALIB_DIR = os.path.join(
    os.path.dirname(pyFAI.__file__), "resources", "calibration"
)

# Units the verifier checks via get_peaks. Names mirror rsfai_calibrant::PeakUnit.
PEAK_UNITS = ["2th_deg", "2th_rad", "q_nm^-1", "q_A^-1"]

# (calibrant name, list of wavelengths in meters). The LaB6 wavelengths exercise
# the asin-split boundary (more/fewer visible rings; cf. test_calibrant.py:test_2th
# expecting 25 / 59 / 15 rings at 1.54e-10 / 1e-10 / 2e-10).
DOT_D_CASES = [
    ("AgBh", [1e-10, 1.54e-10]),
    ("Al", [1e-10, 0.7e-10]),
    ("LaB6", [1.54e-10, 1e-10, 2e-10]),
    ("Si", [1e-10, 1.2e-10]),
    ("CeO2", [1e-10, 1.54e-10]),
]

# (name, Cell factory thunk, dmin). These reproduce the lattice -> d-spacing
# arithmetic the shipped .D files were generated from.
def _cells():
    # Cr2O3 (eskolaite), R-3c / space group 167: primitive hexagonal cell with
    # the c-glide reflection condition appended, exactly as
    # doc/.../Calibrant/new_calibrant.ipynb builds it.
    cr2o3 = Cell.hexagonal(4.958979, 13.59592)
    cr2o3.selection_rules.append(ReflectionCondition.group167_R3bar_c)
    # Hydrocerussite, R-3m / space group 166 (Calibrant/hydrocerussite.ipynb).
    hydroc = Cell.hexagonal(5.24656, 23.7023)
    hydroc.selection_rules.append(ReflectionCondition.group166_R3bar_m)
    return [
        ("Al_cubic_F", Cell.cubic(4.0495, lattice_type="F"), 1.0),
        ("LaB6_cubic_P", Cell.cubic(4.1568, lattice_type="P"), 1.0),
        ("Si_diamond", Cell.diamond(5.4312), 1.0),
        ("CeO2_cubic_F", Cell.cubic(5.411651, lattice_type="F"), 1.0),
        ("Cr2O3_R3c_167", cr2o3, 1.0),
        ("hydrocerussite_R3m_166", hydroc, 1.0),
    ]


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name), arr)
    return list(arr.shape), str(arr.dtype)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    dot_d_meta = []
    for name, wavelengths in DOT_D_CASES:
        # Copy the shipped .D file so the verifier is self-contained / offline.
        src = os.path.join(CALIB_DIR, f"{name}.D")
        dst = os.path.join(OUTDIR, f"{name}.D")
        shutil.copyfile(src, dst)

        wl_meta = []
        for wl in wavelengths:
            cal = get_calibrant(name)
            cal.wavelength = wl  # triggers _calc_2th
            tag = f"{name}__wl{wl:.3e}"

            dsp = np.asarray(cal.dspacing, dtype=np.float64)
            tth = np.asarray(cal.get_2th(), dtype=np.float64)
            f = {
                "dspacing": save(f"{tag}__dspacing.npy", dsp),
                "two_theta": save(f"{tag}__two_theta.npy", tth),
            }
            peaks = {}
            for unit in PEAK_UNITS:
                pk = np.asarray(cal.get_peaks(unit), dtype=np.float64)
                key = unit.replace("^", "").replace("-1", "m1")
                peaks[unit] = save(f"{tag}__peaks_{key}.npy", pk)
            f["peaks"] = peaks
            wl_meta.append(
                {
                    "wavelength": wl,
                    "tag": tag,
                    "n_rings": int(tth.shape[0]),
                    "n_dspacing": int(dsp.shape[0]),
                    "fields": f,
                }
            )
        dot_d_meta.append({"name": name, "file": f"{name}.D", "cases": wl_meta})

    cell_meta = []
    for tag, cell, dmin in _cells():
        groups = cell.calculate_dspacing(dmin)
        # Keys descending, exactly as build_calibrant_config orders them.
        keys = sorted(groups.keys(), reverse=True)
        dsp = np.asarray(keys, dtype=np.float64)
        mult = np.asarray([len(groups[k]) for k in keys], dtype=np.int32)
        cell_meta.append(
            {
                "tag": tag,
                "dmin": dmin,
                "n_dspacing": int(dsp.shape[0]),
                "fields": {
                    "dspacing": save(f"cell_{tag}__dspacing.npy", dsp),
                    "multiplicity": save(f"cell_{tag}__multiplicity.npy", mult),
                },
            }
        )

    manifest = {
        "dataset": "calibrant",
        "source": "gen_golden_calibrant.py",
        "pyfai_version": pyFAI.version,
        "numpy_version": np.__version__,
        "const_hc": float(CONST_hc),
        "const_hc_hex": float(CONST_hc).hex(),
        "peak_units": PEAK_UNITS,
        "bragg_formula": "tth = 2 * asin(5e9 * wavelength / d)",
        "dot_d": dot_d_meta,
        "cells": cell_meta,
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as fd:
        json.dump(manifest, fd, indent=2)

    print(f"wrote calibrant golden to {OUTDIR}")
    print(f"  CONST_hc = {float(CONST_hc)!r}")
    for c in dot_d_meta:
        rings = ", ".join(f"{w['wavelength']:.2e}:{w['n_rings']}" for w in c["cases"])
        print(f"  {c['name']}: rings per wavelength = {rings}")
    for c in cell_meta:
        print(f"  cell {c['tag']}: {c['n_dspacing']} d-spacings (dmin={c['dmin']})")


if __name__ == "__main__":
    main()
