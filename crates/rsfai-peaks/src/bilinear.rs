//! Bilinear peak interpolator, a port of `pyFAI/ext/bilinear.pxi`'s `Bilinear`.
//!
//! Two operations are used by the peak finders:
//!   * [`Bilinear::local_maxi_index`] (`c_local_maxi`, `bilinear.pxi:215`): a
//!     discrete 3x3 hill-climb that returns the flat index of the local maximum
//!     reachable from a start pixel. Drives the inverse-watershed labelling.
//!   * [`Bilinear::local_maxi`] (`local_maxi`, `bilinear.pxi:147`): sub-pixel
//!     refinement via a second-order Taylor expansion of the discrete maximum,
//!     falling back to a 3x3 centre-of-mass. Returns `(y, x)` as `f32`.
//!
//! All arithmetic is `f32`, matching the Cython `float` storage and operations,
//! so the refined coordinates are reproducible bit-for-bit.

/// A 2-D `f32` image wrapped so peaks can be located like a continuous function.
pub struct Bilinear<'a> {
    data: &'a [f32],
    pub height: usize,
    pub width: usize,
}

impl<'a> Bilinear<'a> {
    /// Wrap a row-major `f32` image of shape `(height, width)`.
    pub fn new(data: &'a [f32], height: usize, width: usize) -> Self {
        assert_eq!(
            data.len(),
            height * width,
            "data length must be height*width"
        );
        Bilinear {
            data,
            height,
            width,
        }
    }

    #[inline]
    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.width + c]
    }

    /// Discrete hill-climb local maximum (`c_local_maxi`, `bilinear.pxi:215`):
    /// from flat index `x`, repeatedly step to the largest value in the 3x3
    /// neighbourhood until no neighbour is strictly larger. Returns the flat
    /// index of the maximum.
    pub fn local_maxi_index(&self, x: usize) -> usize {
        let mut current0 = (x / self.width) as i64;
        let mut current1 = (x % self.width) as i64;
        let mut value = self.at(current0 as usize, current1 as usize);
        let mut old_value = f32::NEG_INFINITY;
        let (mut new0, mut new1) = (current0, current1);

        while value > old_value {
            old_value = value;
            let start0 = (current0 - 1).max(0);
            let stop0 = (current0 + 2).min(self.height as i64);
            let start1 = (current1 - 1).max(0);
            let stop1 = (current1 + 2).min(self.width as i64);
            for i0 in start0..stop0 {
                for i1 in start1..stop1 {
                    let tmp = self.at(i0 as usize, i1 as usize);
                    if tmp > value {
                        new0 = i0;
                        new1 = i1;
                        value = tmp;
                    }
                }
            }
            current0 = new0;
            current1 = new1;
        }
        (self.width as i64 * current0 + current1) as usize
    }

    /// Sub-pixel local maximum (`local_maxi`, `bilinear.pxi:147`). `x` is a
    /// `(y, x)` integer-ish seed; the value is rounded to the nearest pixel,
    /// hill-climbed, then refined by a second-order Taylor expansion (or a 3x3
    /// centre of mass when the Hessian is singular / the step exceeds one
    /// pixel). Returns the refined `(y, x)` as `f32`.
    pub fn local_maxi(&self, x: (f32, f32)) -> (f32, f32) {
        // round(x[0]) * width + round(x[1]); Cython `round` on a Python float
        // is banker's rounding, but here the inputs are the integer peak coords
        // produced by the watershed, so round-half is never exercised. Match
        // Python's round-half-to-even to be safe.
        let r0 = round_half_even(x.0 as f64) as i64;
        let r1 = round_half_even(x.1 as f64) as i64;
        let seed = (r0 * self.width as i64 + r1).max(0) as usize;
        let res = self.local_maxi_index(seed);
        let current0 = (res / self.width) as i64;
        let current1 = (res % self.width) as i64;

        if current0 > 0
            && current0 < self.height as i64 - 1
            && current1 > 0
            && current1 < self.width as i64 - 1
        {
            let c0 = current0 as usize;
            let c1 = current1 as usize;
            let a00 = self.at(c0 - 1, c1 - 1);
            let a01 = self.at(c0 - 1, c1);
            let a02 = self.at(c0 - 1, c1 + 1);
            let a10 = self.at(c0, c1 - 1);
            let a11 = self.at(c0, c1);
            let a12 = self.at(c0, c1 + 1);
            let a20 = self.at(c0 + 1, c1 - 1);
            let a21 = self.at(c0 + 1, c1);
            // NOTE: pyFAI uses `a22 = self.data[current0 + 1, current1 - 1]`
            // (a copy of a20, an upstream quirk at bilinear.pxi:182). Reproduce
            // it verbatim so the refinement matches bit-for-bit.
            let a22 = self.at(c0 + 1, c1 - 1);
            let d00 = a12 - 2.0 * a11 + a10;
            let d11 = a21 - 2.0 * a11 + a01;
            let d01 = (a00 - a02 - a20 + a22) / 4.0;
            let denom = 2.0 * (d00 * d11 - d01 * d01);
            if denom.abs() >= 1e-10 {
                let delta0 = ((a12 - a10) * d01 + (a01 - a21) * d11) / denom;
                let delta1 = ((a10 - a12) * d00 + (a21 - a01) * d01) / denom;
                if delta0.abs() <= 1.0 && delta1.abs() <= 1.0 {
                    return (delta0 + current0 as f32, delta1 + current1 as f32);
                }
            }
            // centre-of-mass fallback over the 3x3 patch.
            let mut sum0 = 0f32;
            let mut sum1 = 0f32;
            let mut sum = 0f32;
            for i0 in (c0 - 1)..=(c0 + 1) {
                for i1 in (c1 - 1)..=(c1 + 1) {
                    let tmp = self.at(i0, i1);
                    sum0 += tmp * i0 as f32;
                    sum1 += tmp * i1 as f32;
                    sum += tmp;
                }
            }
            if sum > 0.0 {
                return (sum0 / sum, sum1 / sum);
            }
        }
        (current0 as f32, current1 as f32)
    }
}

/// Round half to even (Python 3 / numpy convention), in f64 to match CPython's
/// `round` on the float coordinates pyFAI passes in.
#[inline]
fn round_half_even(x: f64) -> f64 {
    let r = x.round(); // round half away from zero
    if (x - x.trunc()).abs() == 0.5 {
        // exactly halfway: pick the even neighbour
        let floor = x.floor();
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        r
    }
}
