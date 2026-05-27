# rsfai-py

PyO3 drop-in module, built with [maturin], exposing the bit-exact rsFAI
integration kernels to Python as the importable module `rsfai`. Its purpose is
**in-process side-by-side validation**: a test running in the `daq` conda env
imports both `pyFAI` and `rsfai` and compares `numpy.ndarray.tobytes()` on
identical input arrays.

This crate is intentionally **excluded** from the default Cargo workspace (see
the root `Cargo.toml`) so `cargo build` / `cargo test` of the engines do not
require the extension-module linkage. It is a `cdylib` built standalone by
maturin; its path dependencies (`rsfai-core`, `rsfai-integrate`, `rsfai-preproc`)
still resolve their workspace inheritance from the root.

## Exposed kernels

numpy-in / numpy-out, no arithmetic of its own — every value comes from the
already-validated Rust engines:

- `preproc4` — per-pixel preprocessing → `(npix, 4)` `[signal, variance, norm, count]`
- `histogram_preproc`, `histogram1d`, `histogram2d` — no-split binning
- `build_bbox_csr_1d` / `build_bbox_csr_2d`, `build_full_csr_1d` / `build_full_csr_2d` — CSR build
- `csr_integrate1d` / `csr_integrate2d` — CSR apply

Preprocessed rows are passed as `(npix, 4)` f32; corner arrays are passed
pre-flattened to f64 (`(npix*4*2,)`). See the module docstring in `src/lib.rs`
for the per-function signatures.

## Build into the daq env

```sh
cd crates/rsfai-py
env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \
    /Users/stevek/mamba/envs/daq/bin/maturin develop --release
```

## Run the parity test

```sh
OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/test_inprocess_parity.py
```

It asserts, for every committed golden dataset, that `rsfai == committed golden`
**and** `live pyFAI == committed golden` (so `rsfai == live pyFAI`) for every
output field, bit-for-bit. Divergence is reported as a max-ULP figure and fails
the test — tolerance is never silently widened.

[maturin]: https://github.com/PyO3/maturin
