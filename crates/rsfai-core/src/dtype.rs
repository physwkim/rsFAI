//! The pyFAI dtype contract, ported verbatim from
//! `src/pyFAI/ext/regrid_common.pxi:56-78`.
//!
//! Bit-exact parity requires every array to use the same width pyFAI uses:
//! positions in f64, image/weights/coefficients in f32, accumulators in f64.
//! Do not "promote for safety" — a wider accumulator changes the rounding and
//! breaks Tier A (see `doc/bit-exact-ladder.md`).

/// Positions: `pos0`, `pos1`, deltas, bin edges. pyFAI `position_t`.
pub type PositionT = f64;
/// Weights / image / sparse coefficients. pyFAI `data_t`.
pub type DataT = f32;
/// Accumulators (signal, variance, norm, count, norm²). pyFAI `acc_t`.
pub type AccT = f64;
/// Mask. pyFAI `mask_t` (0 = valid, non-zero = masked).
pub type MaskT = i8;
/// Sparse-matrix / bin indices. pyFAI `index_t`.
pub type IndexT = i32;
/// Pixel-splitting work buffers. pyFAI `buffer_t`.
pub type BufferT = f32;

/// `EPS32 = 1.0 + f32::EPSILON`, evaluated in f64 — matches
/// `regrid_common.pxi:117` (`1.0 + numpy.finfo(numpy.float32).eps`). Used by
/// [`calc_upper_bound`].
pub const EPS32: f64 = 1.0 + f32::EPSILON as f64;

/// Smallest f32-resolution-greater upper bound for a histogram, matching
/// `calc_upper_bound` in `regrid_common.pxi:138-146`.
#[inline]
pub fn calc_upper_bound(maximum_value: f64) -> f64 {
    if maximum_value > 0.0 {
        maximum_value * EPS32
    } else {
        maximum_value / EPS32
    }
}

/// One sparse-matrix entry: `lut_t { idx: i32, coef: f32 }`
/// (`regrid_common.pxi:81-84`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LutEntry {
    pub idx: IndexT,
    pub coef: DataT,
}

/// pyFAI error model, matching the `int error_model` codes used throughout the
/// kernels (`preproc_value_inplace`, `update_1d_accumulator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    /// 0: disabled.
    No,
    /// 1: propagate the supplied per-pixel variance.
    Variance,
    /// 2: Poisson — `variance = max(1.0, data)`.
    Poisson,
    /// 3: azimuthal — Welford-style online variance across the bin.
    Azimuthal,
}

impl ErrorModel {
    /// The integer code pyFAI uses internally.
    #[inline]
    pub fn code(self) -> i32 {
        match self {
            ErrorModel::No => 0,
            ErrorModel::Variance => 1,
            ErrorModel::Poisson => 2,
            ErrorModel::Azimuthal => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps32_matches_python_value() {
        // numpy.finfo(numpy.float32).eps == 2**-23; 1.0 + that, in f64.
        assert_eq!(EPS32, 1.0 + 2f64.powi(-23));
    }

    #[test]
    fn upper_bound_branches() {
        assert!(calc_upper_bound(10.0) > 10.0);
        assert!(calc_upper_bound(-10.0) > -10.0); // /EPS32 moves toward zero
        assert_eq!(calc_upper_bound(0.0), 0.0);
    }
}
