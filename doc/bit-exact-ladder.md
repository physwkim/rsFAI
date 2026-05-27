# The bit-exact ladder

rsFAI's correctness target is **bitwise-identical output to pyFAI** wherever that
is physically constructible. End-to-end bitwise identity from raw image →
integrated curve **cannot be promised by construction**, for two reasons that no
amount of careful coding removes:

1. **Parallel reductions reorder float adds.** pyFAI's `histogram.pyx`
   (`_histogram_omp`) and the CSR/bbox kernels use OpenMP `prange`; per-thread
   partial sums merge in a nondeterministic order. IEEE-754 `+` is not
   associative, so the sum is not reproducible bit-for-bit across thread counts.
2. **libm transcendentals are not correctly-rounded.** Geometry uses
   `sin`/`cos`/`atan2` (`ext/_geometry.pyx` via `libc.math`; numpy's vectorized
   path uses a different SIMD libm). `+ - * / sqrt` are IEEE-754
   correctly-rounded and therefore reproducible; `sin`/`cos`/`atan2`/`exp` are
   not, and differ between numpy's SIMD libm, the C libm pyFAI's Cython links,
   and Rust's libm.

A third boundary is **eliminated by construction** rather than tolerated:

3. **FMA contraction.** A C compiler with `-ffp-contract=on`/`fast` (clang's
   default on arm64) fuses `a*b + c` into a single `fma(a, b, c)` with one
   rounding instead of two — so even pure `+ - *` algebra (e.g. `calc_pos_zyx`'s
   rotation polynomial) diverges from Rust's non-fused `+ - *` at the last bit
   (~1e-16 absolute; up to ~128 ULP near zero-crossings where the magnitude is
   tiny). This is **not** an unavoidable physical limit: it is removed by
   compiling pyFAI with `-ffp-contract=off`, after which both sides evaluate the
   bare IEEE-754 expression and the algebra is bitwise-identical. See the
   provenance note below; measured result in Tier B.

So "bit-exact" is delivered as a **staged ladder**, with boundaries 1–2 pinned
and boundary 3 removed at the source.

## Pinning (applies to all tiers)

- Golden data is generated with **`OMP_NUM_THREADS=1`** and pyFAI's serial /
  Cython code path (not OpenCL).
- Every golden dataset records provenance in `manifest.json`: pyFAI version,
  numpy version, platform, `OMP_NUM_THREADS`, the method tuple, and the dtype of
  each array.
- The Rust default test path is **serial**; `rayon` is opt-in behind a feature
  flag and is never the bit-exact gate — only ever checked at tolerance.
- **No-FMA source build.** Golden is generated from pyFAI 2026.5.0 **rebuilt
  from the local `~/codes/pyFAI` source** into the `daq` env with FMA
  contraction disabled:

  ```sh
  # build deps into daq (numpy already present, so --no-build-isolation)
  daq/bin/python -m pip install "meson-python>=0.11" "meson>=1.1" ninja wheel \
      "Cython>=0.29.31" "pyproject-metadata>=0.5.0"
  # build a no-FMA wheel from the local source tree (clang on arm64)
  cd ~/codes/pyFAI
  PATH="$(dirname daq/bin/python):$PATH" \
    CFLAGS="-ffp-contract=off" CXXFLAGS="-ffp-contract=off" \
    daq/bin/python -m pip wheel . --no-build-isolation --no-deps -w /tmp/pyfai-nofma
  daq/bin/python -m pip install --force-reinstall --no-deps /tmp/pyfai-nofma/pyfai-*.whl
  ```

  This links Apple's system libm (the same libm Rust's `std` f64 transcendentals
  call) **and** removes FMA fusion, so the algebraic transform is bitwise-exact
  by construction (boundary 3 above). The `manifest.json` `build` block records
  `cflags`/`source_tree`. Tier A is libm- and FMA-independent regardless.

## dtype contract

Ported verbatim from `pyFAI/ext/regrid_common.pxi:56-78`. See
`crates/rsfai-core/src/dtype.rs`.

| pyFAI ctype | meaning | Rust type |
|---|---|---|
| `position_t` | positions: `pos0`/`pos1`, deltas, bin edges | `f64` |
| `data_t` | weights / image / coefficients | `f32` |
| `acc_t` | accumulators (signal, variance, norm, count, norm²) | `f64` |
| `mask_t` | mask (0 = valid) | `i8` |
| `index_t` | sparse / bin indices | `i32` |
| `buffer_t` | split work buffers | `f32` |

Do **not** widen accumulators "for safety": a wider type changes the rounding
and breaks Tier A.

## Tiers

### Tier A — integration kernels (REQUIRE bitwise-exact)

Feed the Rust kernels the *identical* inputs pyFAI used, dumped to disk:
position arrays (`pos0`, `delta_pos0`, `pos1`, `delta_pos1`), per-pixel preproc
output (`signal`, `variance`, `norm`, `count`), and — for CSR — the sparse
matrix (`data`/`indices`/`indptr`). With identical inputs, identical dtypes, and
identical accumulation order, the histogram/bbox/CSR/full-split outputs **must
match every bit**. This is a true test of the binning + accumulation logic,
independent of libm. **This is the correctness gate.**

### Tier B — geometry & preproc (exact for algebra, ULP-budgeted for transcendentals)

Algebraic arrays (only `+ - * / sqrt`) must be bitwise-exact vs golden.
Transcendental-derived arrays (`tth`, `chi`, `q`) must match exactly **if** the
libm agrees; otherwise the divergence is measured in ULPs, the budget is
recorded in the manifest (`ulp_budget`), and any pixel whose bin assignment flips
at a bin boundary because of it is enumerated. Tolerance is never silently
widened — the ULP delta is reported as an explicit, tracked number.

**Measured (M1, Pilatus1M, 1,023,183 pixels, no-FMA build):**

| array | math | max_ulp | result |
|---|---|---|---|
| `calc_pos_zyx` z/y/x | `+ - *` only | 0 | bitwise-exact |
| `pos0_center` (`q_nm⁻¹`, `2th_deg`) | `sqrt` + `atan2`/`sin` | 0 | bitwise-exact |
| `chi_center` | `atan2` | 0 | bitwise-exact |

The transcendental arrays measured **0 ULP** — on this machine numexpr's libm
and Rust's `std` libm agree bit-for-bit — so the geometry test asserts them
bit-exact (`is_bit_exact`), not at a budget. The test still prints `max_ulp`, so
a future libm divergence fails loudly and is then recorded as a manifest budget.

### Tier C — full pipeline (raw image → curve)

Bitwise-exact is the *target*, achieved whenever Tier B is bitwise-exact
(matching libm). When Tier B diverges by N ULP, Tier C inherits a bounded,
documented divergence. Tier C is the integration test; **Tier A is the gate.**

## The arithmetic to reproduce (from `regrid_common.pxi`)

- **preproc** (`preproc_value_inplace`, lines 149-237): `signal = data - dark`;
  `norm = normalization_factor * flat * polarization * solidangle * absorption`;
  Poisson (`error_model == 2`): `variance = max(1.0, data)`; invalid pixel
  (mask/NaN/dummy/`norm == 0`) → all four outputs zero.
- **1D accumulate** (`update_1d_accumulator`, lines 240-301): `w = weight*norm`;
  `sum_sig += signal*weight`; `sum_var += variance*weight²`; `sum_nrm += w`;
  `sum_nrm2 += w²`; `sum_cnt += count*weight`. `error_model == 3` (azimuthal)
  uses the Welford-style online update at lines 265-287 — port exactly.
- **2D accumulate**: `update_2d_accumulator`, lines 304-322.
- **bin number**: `get_bin_number` = `(x0 - pos0_min) / delta`; histogram upper
  bound via `calc_upper_bound` using `EPS32 = 1.0 + f32::EPSILON`.
