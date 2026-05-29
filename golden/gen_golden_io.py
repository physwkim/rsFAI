#!/usr/bin/env python
"""Golden generator for rsfai-io (HDF5 / NeXus reading via the pure-Rust
`rust-hdf5` crate).

Writes a small NeXus-shaped HDF5 file with h5py (libhdf5), plus the .npy of
every dataset, so the Rust verifier can prove `rust-hdf5` reads an
h5py/libhdf5-written file back BIT-EXACTLY across the rsFAI dtypes
(f32 image, i32 counts, f64 positions). One dataset is stored gzip-compressed +
chunked to exercise `rust-hdf5`'s filter pipeline; the rest are contiguous.

Run in the daq env (h5py 3.16, libhdf5 2.0):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        /Users/stevek/mamba/envs/daq/bin/python golden/gen_golden_io.py
"""

import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_io")

# NeXus dataset paths the Rust verifier reads.
P_DATA = "entry/data/data"
P_DATA_GZIP = "entry/data/data_gzip"
P_COUNTS = "entry/data/counts"
P_POS = "entry/instrument/detector/positions"


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Deterministic, varied arrays (no NaN/inf → exact compare on the Rust side).
    idx = np.arange(32 * 32)
    data = ((idx % 97).astype(np.float32) * 1.5 + 0.25).reshape(32, 32)
    counts = ((idx * 7 - 3).astype(np.int32)).reshape(32, 32)
    positions = np.linspace(-5.0, 12.5, 64).astype(np.float64)

    h5path = os.path.join(OUTDIR, "frame.h5")
    with h5py.File(h5path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"

        nxdata = entry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "data"

        # Contiguous (default) f32 image + i32 counts.
        nxdata.create_dataset("data", data=data)
        nxdata.create_dataset("counts", data=counts)
        # Same image, gzip-compressed + chunked: exercises the filter pipeline.
        nxdata.create_dataset(
            "data_gzip", data=data, chunks=(8, 8), compression="gzip", compression_opts=4
        )

        det = entry.create_group("instrument").create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset("positions", data=positions)

    np.save(os.path.join(OUTDIR, "frame__data_f32.npy"), data)
    np.save(os.path.join(OUTDIR, "frame__counts_i32.npy"), counts)
    np.save(os.path.join(OUTDIR, "frame__positions_f64.npy"), positions)

    manifest = {
        "h5py_version": h5py.__version__,
        "hdf5_version": h5py.version.hdf5_version,
        "numpy_version": np.__version__,
        "file": "frame.h5",
        "datasets": {
            P_DATA: {"shape": list(data.shape), "dtype": "float32", "layout": "contiguous"},
            P_DATA_GZIP: {"shape": list(data.shape), "dtype": "float32", "layout": "gzip/chunked"},
            P_COUNTS: {"shape": list(counts.shape), "dtype": "int32", "layout": "contiguous"},
            P_POS: {"shape": list(positions.shape), "dtype": "float64", "layout": "contiguous"},
        },
    }
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote HDF5/NeXus golden to {OUTDIR}")
    print(f"  {h5path}  (h5py {h5py.__version__}, libhdf5 {h5py.version.hdf5_version})")
    print(f"  datasets: {P_DATA} (f32 {data.shape}), {P_DATA_GZIP} (gzip), "
          f"{P_COUNTS} (i32), {P_POS} (f64 {positions.shape})")


if __name__ == "__main__":
    main()
