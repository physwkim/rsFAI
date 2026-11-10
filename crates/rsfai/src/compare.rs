//! Bit-pattern and ULP comparison against golden arrays.
//!
//! Tier A (integration kernels fed identical inputs) and algebraic Tier B
//! require **bitwise** equality: we compare `f64::to_bits` / `f32::to_bits`, not
//! `==`, so that `-0.0` vs `+0.0` and distinct NaN payloads are caught. For
//! transcendental Tier B we also report the worst-case ULP distance and the
//! number of bin-boundary flips, which the caller asserts against a recorded
//! budget. See `doc/bit-exact-ladder.md`.

/// Outcome of comparing an actual array against a golden array.
#[derive(Debug, Clone)]
pub struct CompareReport {
    /// Number of elements compared.
    pub total: usize,
    /// Elements whose raw bit pattern differs.
    pub bit_mismatches: usize,
    /// Elements where exactly one side is NaN (a real disagreement).
    pub nan_only_one_side: usize,
    /// Worst-case ULP distance over finite, same-sign-of-NaN elements.
    pub max_ulp: u64,
    /// Worst-case absolute difference over finite elements.
    pub max_abs_diff: f64,
    /// Worst-case relative difference `|actual - golden| / |golden|` over finite
    /// elements. When `golden == 0` the element is exact iff `actual == 0`
    /// (relative error is then `0`); a non-zero `actual` against a zero `golden`
    /// yields `+inf`, failing any finite tolerance. Used by [`within_rel`] for
    /// the non-bit-reproducible parallel histogram reduction (the f64 reorder
    /// error is bounded by `~n·eps`, far under the 1e-6 gate).
    ///
    /// [`within_rel`]: CompareReport::within_rel
    pub max_rel_diff: f64,
    /// First differing element: `(flat_index, actual, golden)`.
    pub first_mismatch: Option<(usize, f64, f64)>,
}

impl CompareReport {
    /// True iff every element matched bit-for-bit.
    #[inline]
    pub fn is_bit_exact(&self) -> bool {
        self.bit_mismatches == 0 && self.nan_only_one_side == 0
    }

    /// True iff every element is within `budget` ULPs (and no one-sided NaNs).
    #[inline]
    pub fn within_ulp(&self, budget: u64) -> bool {
        self.max_ulp <= budget && self.nan_only_one_side == 0
    }

    /// True iff every element is within `tol` relative error (and no one-sided
    /// NaNs). The pass criterion for parallel-histogram reduction outputs, whose
    /// f64 accumulation order is non-deterministic: see [`max_rel_diff`].
    ///
    /// [`max_rel_diff`]: CompareReport::max_rel_diff
    #[inline]
    pub fn within_rel(&self, tol: f64) -> bool {
        self.max_rel_diff <= tol && self.nan_only_one_side == 0
    }
}

/// Per-element relative difference `|a - g| / |g|`, with the zero-golden
/// convention documented on [`CompareReport::max_rel_diff`]. Operands are passed
/// as f64 (f32 callers widen first).
#[inline]
fn rel_diff(a: f64, g: f64) -> f64 {
    let denom = g.abs();
    if denom == 0.0 {
        if a == g {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (a - g).abs() / denom
    }
}

/// Map an f64 to a monotonic i64 key so that adjacent floats differ by 1
/// (sign-magnitude → two's-complement ordering; the classic
/// "AlmostEqual2sComplement" transform).
#[inline]
fn ulp_key_f64(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    if b < 0 {
        i64::MIN.wrapping_sub(b)
    } else {
        b
    }
}

#[inline]
fn ulp_key_f32(x: f32) -> i32 {
    let b = x.to_bits() as i32;
    if b < 0 {
        i32::MIN.wrapping_sub(b)
    } else {
        b
    }
}

/// Compare two equal-length f64 slices.
pub fn compare_f64(actual: &[f64], golden: &[f64]) -> CompareReport {
    assert_eq!(
        actual.len(),
        golden.len(),
        "length mismatch: actual {} vs golden {}",
        actual.len(),
        golden.len()
    );
    let mut report = CompareReport {
        total: actual.len(),
        bit_mismatches: 0,
        nan_only_one_side: 0,
        max_ulp: 0,
        max_abs_diff: 0.0,
        max_rel_diff: 0.0,
        first_mismatch: None,
    };
    for (i, (&a, &g)) in actual.iter().zip(golden.iter()).enumerate() {
        if a.to_bits() == g.to_bits() {
            continue;
        }
        report.bit_mismatches += 1;
        if report.first_mismatch.is_none() {
            report.first_mismatch = Some((i, a, g));
        }
        if a.is_nan() ^ g.is_nan() {
            report.nan_only_one_side += 1;
            continue; // ULP/abs undefined when only one side is NaN
        }
        if a.is_nan() && g.is_nan() {
            continue; // both NaN, payload differs — counted as bit mismatch only
        }
        let ulp = ulp_key_f64(a).wrapping_sub(ulp_key_f64(g)).unsigned_abs();
        report.max_ulp = report.max_ulp.max(ulp);
        let abs = (a - g).abs();
        if abs > report.max_abs_diff {
            report.max_abs_diff = abs;
        }
        let rel = rel_diff(a, g);
        if rel > report.max_rel_diff {
            report.max_rel_diff = rel;
        }
    }
    report
}

/// Compare two equal-length f32 slices. ULP distance is computed in f32 width
/// and reported as `u64`.
pub fn compare_f32(actual: &[f32], golden: &[f32]) -> CompareReport {
    assert_eq!(
        actual.len(),
        golden.len(),
        "length mismatch: actual {} vs golden {}",
        actual.len(),
        golden.len()
    );
    let mut report = CompareReport {
        total: actual.len(),
        bit_mismatches: 0,
        nan_only_one_side: 0,
        max_ulp: 0,
        max_abs_diff: 0.0,
        max_rel_diff: 0.0,
        first_mismatch: None,
    };
    for (i, (&a, &g)) in actual.iter().zip(golden.iter()).enumerate() {
        if a.to_bits() == g.to_bits() {
            continue;
        }
        report.bit_mismatches += 1;
        if report.first_mismatch.is_none() {
            report.first_mismatch = Some((i, a as f64, g as f64));
        }
        if a.is_nan() ^ g.is_nan() {
            report.nan_only_one_side += 1;
            continue;
        }
        if a.is_nan() && g.is_nan() {
            continue;
        }
        let ulp = ulp_key_f32(a).wrapping_sub(ulp_key_f32(g)).unsigned_abs() as u64;
        report.max_ulp = report.max_ulp.max(ulp);
        let abs = (a as f64 - g as f64).abs();
        if abs > report.max_abs_diff {
            report.max_abs_diff = abs;
        }
        // Relative error in f64 (the f32 operands are widened, matching abs).
        let rel = rel_diff(a as f64, g as f64);
        if rel > report.max_rel_diff {
            report.max_rel_diff = rel;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_bit_exact() {
        let a = [1.0_f64, -2.5, 0.0, f64::INFINITY];
        let r = compare_f64(&a, &a);
        assert!(r.is_bit_exact());
        assert_eq!(r.max_ulp, 0);
    }

    #[test]
    fn signed_zero_is_a_bit_mismatch() {
        let r = compare_f64(&[0.0], &[-0.0]);
        assert!(!r.is_bit_exact());
        assert_eq!(r.bit_mismatches, 1);
    }

    #[test]
    fn one_ulp_distance() {
        let x = 1.0_f64;
        let next = f64::from_bits(x.to_bits() + 1);
        let r = compare_f64(&[next], &[x]);
        assert_eq!(r.max_ulp, 1);
        assert!(!r.is_bit_exact());
        assert!(r.within_ulp(1));
    }

    #[test]
    fn one_sided_nan_flagged() {
        let r = compare_f64(&[f64::NAN], &[1.0]);
        assert_eq!(r.nan_only_one_side, 1);
        assert!(!r.within_ulp(u64::MAX));
    }

    #[test]
    fn f32_one_ulp() {
        let x = 3.5_f32;
        let next = f32::from_bits(x.to_bits() + 1);
        let r = compare_f32(&[next], &[x]);
        assert_eq!(r.max_ulp, 1);
    }

    #[test]
    fn relative_diff_within_tolerance() {
        // 1e6 vs (1e6 + 0.5): relative error 5e-7, under a 1e-6 gate but not
        // bit-exact — the shape of a parallel-reduction reorder difference.
        let g = 1.0e6_f64;
        let a = g + 0.5;
        let r = compare_f64(&[a], &[g]);
        assert!(!r.is_bit_exact());
        assert!(r.within_rel(1e-6));
        assert!(!r.within_rel(1e-7));
    }

    #[test]
    fn zero_golden_requires_zero_actual() {
        // Empty histogram bins are exactly 0 on both sides -> relative error 0.
        let r0 = compare_f64(&[0.0], &[0.0]);
        assert!(r0.within_rel(1e-6));
        // A non-zero actual against a zero golden is an infinite relative error.
        let r1 = compare_f64(&[1e-12], &[0.0]);
        assert!(!r1.within_rel(1e-6));
        assert_eq!(r1.max_rel_diff, f64::INFINITY);
    }
}
