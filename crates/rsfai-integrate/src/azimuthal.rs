//! Shared azimuthal-error-model (Welford) accumulation step.
//!
//! pyFAI's `error_model="azimuthal"` does not propagate a per-pixel variance.
//! Instead, for each output bin it estimates the variance of the pixel
//! intensities `b = signal/norm` falling in that bin, accumulated online with a
//! weighted Welford update:
//!
//! ```text
//! VV_{A∪b} = VV_A + ω_b² · (b − ⟨A⟩) · (b − ⟨A∪b⟩)
//! ```
//!
//! The identical update appears in `regrid_common.pxi:265`
//! (`update_1d_accumulator`), `histogram.pyx` (`histogram_preproc`), and the
//! `do_azimuthal_variance` blocks of `CSR_common.pxi`, `CSC_common.pxi`, and
//! `LUT_common.pxi`. Every engine reduces to the same f64 arithmetic on
//! `(sum_sig, sum_var, sum_norm, sum_norm_sq)`; only the *inputs* differ in how
//! they are formed, so the caller computes them:
//!
//! - `omega_b = coef·norm` — the contribution's weight (`coef` is the split/LUT
//!   coefficient, `1` for the no-split histogram).
//! - `sig_inc = coef·signal` — the signal increment.
//! - `b = signal/norm` — the per-pixel intensity. CSR/LUT/histogram read
//!   `signal`/`norm` as f64 and divide in f64; CSC and the direct-split
//!   histogram (`update_1d_accumulator`) divide the f32 `value.signal`/
//!   `value.norm` in f32 and promote, so `b` is the caller's responsibility.
//!
//! The first contribution to a bin (`*sum_norm_sq <= 0`) seeds the accumulators;
//! later ones run the Welford update. The per-engine `norm != 0` guard is **not**
//! applied here: pyFAI applies it differently across engines (CSR, the no-split
//! histogram, and the direct-split histogram skip a zero-norm contribution that
//! is not the bin's first; LUT and CSC do not), so the caller decides whether to
//! call this at all.

use rsfai_core::dtype::AccT;

/// One azimuthal (Welford) accumulation step. See the module docs for how the
/// caller forms `omega_b`, `sig_inc`, and `b`, and for the `norm != 0` guard.
#[inline]
pub(crate) fn azimuthal_step(
    sum_sig: &mut AccT,
    sum_var: &mut AccT,
    sum_norm: &mut AccT,
    sum_norm_sq: &mut AccT,
    omega_b: AccT,
    sig_inc: AccT,
    b: AccT,
) {
    if *sum_norm_sq <= 0.0 {
        // First contribution to this bin: seed the accumulators (pyFAI sets
        // sum_norm_sq = (coef·norm)², i.e. omega_b², leaving sum_var at 0).
        *sum_sig = sig_inc;
        *sum_norm = omega_b;
        *sum_norm_sq = omega_b * omega_b;
    } else {
        let omega_a = *sum_norm;
        let omega2_b = omega_b * omega_b;
        *sum_norm = omega_a + omega_b;
        *sum_norm_sq += omega2_b;
        let delta1 = *sum_sig / omega_a - b;
        *sum_sig += sig_inc;
        let delta2 = *sum_sig / *sum_norm - b;
        *sum_var += omega2_b * delta1 * delta2;
    }
}
