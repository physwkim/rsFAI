//! Port of `pyFAI/ext/preproc.pyx::c4_preproc` (split_result = 4).
//!
//! All arithmetic is `f32`, matching `preproc(dtype=float32)`. The output is a
//! flat row-major `f32` array of `4 * size` values, `[signal, variance,
//! normalization, count]` per pixel — the layout of pyFAI's `(size, 4)` result.

/// Optional correction inputs and flags for [`preproc4`], mirroring the
/// `c4_preproc` parameters. All array slices, when present, must have the same
/// length as `data`. Construct via [`PreprocOptions::default`] and set fields.
#[derive(Debug, Clone, Default)]
pub struct PreprocOptions<'a> {
    /// Dark current to subtract from the signal.
    pub dark: Option<&'a [f32]>,
    /// Flat field (multiplies the normalization; also dummy-checked).
    pub flat: Option<&'a [f32]>,
    /// Solid-angle correction (multiplies the normalization). **Pass f32** —
    /// pyFAI casts its f64 solid angle to f32 before preproc.
    pub solidangle: Option<&'a [f32]>,
    /// Polarization correction (multiplies the normalization).
    pub polarization: Option<&'a [f32]>,
    /// Absorption correction (multiplies the normalization).
    pub absorption: Option<&'a [f32]>,
    /// Mask: pixel invalid where `mask[i] != 0`.
    pub mask: Option<&'a [i8]>,
    /// Per-pixel variance (used when not Poisson).
    pub variance: Option<&'a [f32]>,
    /// Dark variance (added to the variance when dark is subtracted).
    pub dark_variance: Option<&'a [f32]>,
    /// Denominator seed; `norm` starts here (pyFAI default 1.0).
    pub normalization_factor: f32,
    /// Poisson error model: `variance = max(data, 1.0)`.
    pub poissonian: bool,
    /// Whether to apply dummy-value masking.
    pub check_dummy: bool,
    /// Dummy (invalid) value.
    pub dummy: f32,
    /// Tolerance around the dummy (`0` ⇒ exact compare).
    pub delta_dummy: f32,
    /// Divide signal/variance by the normalization in place (pyFAI WIP path).
    pub apply_normalization: bool,
}

/// Preprocess `data` (raw signal, f32) into a flat `[signal, variance,
/// normalization, count]`-per-pixel `f32` array of length `4 * data.len()`.
///
/// Bit-exact port of `c4_preproc`; see the module docs for the arithmetic.
pub fn preproc4(data: &[f32], opt: &PreprocOptions) -> Vec<f32> {
    let size = data.len();
    let check = |o: &Option<&[f32]>| {
        if let Some(s) = o {
            assert_eq!(s.len(), size, "correction array length mismatch");
        }
    };
    check(&opt.dark);
    check(&opt.flat);
    check(&opt.solidangle);
    check(&opt.polarization);
    check(&opt.absorption);
    check(&opt.variance);
    check(&opt.dark_variance);
    if let Some(m) = opt.mask {
        assert_eq!(m.len(), size, "mask length mismatch");
    }

    let mut out = Vec::with_capacity(4 * size);
    for i in 0..size {
        let mut one_num = data[i];
        let mut one_den = opt.normalization_factor;
        let mut one_var = if opt.poissonian {
            one_num.max(1.0)
        } else if let Some(v) = opt.variance {
            v[i]
        } else {
            0.0
        };

        let mut is_valid = one_num.is_finite();
        if is_valid {
            if let Some(m) = opt.mask {
                is_valid = m[i] == 0;
            }
        }
        if is_valid && opt.check_dummy {
            is_valid = dummy_ok(one_num, opt.dummy, opt.delta_dummy);
        }
        if is_valid {
            if let Some(flat) = opt.flat {
                is_valid = dummy_ok(flat[i], opt.dummy, opt.delta_dummy);
            }
        }

        let one_count;
        if is_valid {
            if let Some(dark) = opt.dark {
                one_num -= dark[i];
                if let Some(dv) = opt.dark_variance {
                    one_var += dv[i];
                }
            }
            // Order matters for bit-exactness: flat, polarization, solidangle,
            // absorption (c4_preproc lines 450-457).
            if let Some(flat) = opt.flat {
                one_den *= flat[i];
            }
            if let Some(pol) = opt.polarization {
                one_den *= pol[i];
            }
            if let Some(sa) = opt.solidangle {
                one_den *= sa[i];
            }
            if let Some(ab) = opt.absorption {
                one_den *= ab[i];
            }
            if !(one_num.is_finite()
                && one_den.is_finite()
                && one_var.is_finite()
                && one_den != 0.0)
            {
                one_num = 0.0;
                one_var = 0.0;
                one_den = 0.0;
                one_count = 0.0;
            } else {
                one_count = 1.0;
                if opt.apply_normalization {
                    one_num /= one_den;
                    one_var /= one_den * one_den;
                    one_den = 1.0;
                }
            }
        } else {
            one_num = 0.0;
            one_var = 0.0;
            one_den = 0.0;
            one_count = 0.0;
        }

        out.push(one_num);
        out.push(one_var);
        out.push(one_den);
        out.push(one_count);
    }
    out
}

/// Dummy-validity test (`c4_preproc` lines 433-436/440-443): a value is valid
/// when it differs from the dummy (exactly if `delta_dummy == 0`, else by more
/// than `delta_dummy`).
#[inline]
fn dummy_ok(value: f32, dummy: f32, delta_dummy: f32) -> bool {
    if delta_dummy == 0.0 {
        value != dummy
    } else {
        (value - dummy).abs() > delta_dummy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_variance_is_max_data_one() {
        let data = [0.5_f32, 1.0, 4.0];
        let opt = PreprocOptions {
            normalization_factor: 1.0,
            poissonian: true,
            ..Default::default()
        };
        let r = preproc4(&data, &opt);
        // variance = max(data, 1.0): 1.0, 1.0, 4.0
        assert_eq!(r[1], 1.0);
        assert_eq!(r[5], 1.0);
        assert_eq!(r[9], 4.0);
        // signal passes through; norm = 1.0; count = 1.0
        assert_eq!(r[0], 0.5);
        assert_eq!(r[2], 1.0);
        assert_eq!(r[3], 1.0);
    }

    #[test]
    fn masked_and_nonfinite_pixels_are_zeroed() {
        let data = [10.0_f32, f32::NAN, 20.0];
        let mask = [0i8, 0, 1];
        let opt = PreprocOptions {
            mask: Some(&mask),
            normalization_factor: 1.0,
            ..Default::default()
        };
        let r = preproc4(&data, &opt);
        // pixel 0 valid
        assert_eq!(&r[0..4], &[10.0, 0.0, 1.0, 1.0]);
        // pixel 1 NaN -> all zero
        assert_eq!(&r[4..8], &[0.0, 0.0, 0.0, 0.0]);
        // pixel 2 masked -> all zero
        assert_eq!(&r[8..12], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_normalization_invalidates() {
        let data = [10.0_f32];
        let sa = [0.0_f32]; // solid angle 0 -> norm 0 -> invalid
        let opt = PreprocOptions {
            solidangle: Some(&sa),
            normalization_factor: 1.0,
            ..Default::default()
        };
        let r = preproc4(&data, &opt);
        assert_eq!(&r[0..4], &[0.0, 0.0, 0.0, 0.0]);
    }
}
