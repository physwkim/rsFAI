# rsfai-py (deferred to M7)

PyO3 drop-in module, built with [maturin], exposing an `AzimuthalIntegrator`
that mirrors pyFAI's `integrate1d` / `integrate2d` for the supported methods and
units. Its purpose is **in-process side-by-side validation**: a test running in
the `daq` conda env imports both `pyFAI` and `rsfai` and compares
`f64::to_bits()` on identical input arrays (Tier C of the bit-exact ladder).

This crate is intentionally **excluded** from the default Cargo workspace (see
the root `Cargo.toml`) so `cargo build` / `cargo test` do not require the PyO3
extension-module linkage. It is wired up in milestone M7 once the CPU engines
(M1–M6) are bit-exact. See `../../doc/bit-exact-ladder.md` and the plan.

[maturin]: https://github.com/PyO3/maturin
