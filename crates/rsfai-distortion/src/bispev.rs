//! Bivariate B-spline surface evaluation, ported from
//! `pyFAI/ext/_bispev.pyx` (`bisplev` / `cy_bispev` / `fpbspl` / `init_w`),
//! itself a re-implementation of FITPACK's BISPEV.
//!
//! Every value is `f32`: pyFAI casts knots `tx`/`ty`, coefficients `c`, and the
//! evaluation points `x`/`y` to `numpy.float32` before calling `cy_bispev`, and
//! the de Boor–Cox recurrence + the Kahan-summed tensor product run in single
//! precision. To match the bits we keep the identical type width *and* the
//! identical summation order (`i1` outer, `j1` inner, Kahan compensation).

/// A bivariate B-spline tensor representation (`tck` in FITPACK / scipy terms):
/// the knot vectors, the flattened coefficient grid, and the per-axis degrees.
///
/// `c` is indexed `c[ix * nky1 + iy]` with `nky1 = ty.len() - (ky + 1)` — the
/// row-major coefficient layout `cy_bispev` assumes (`l2 = lx*nky1 + ly + ...`).
#[derive(Debug, Clone)]
pub struct Tck {
    /// Knots along the x (first) axis.
    pub tx: Vec<f32>,
    /// Knots along the y (second) axis.
    pub ty: Vec<f32>,
    /// Flattened coefficient grid, length `(tx.len()-kx-1) * (ty.len()-ky-1)`.
    pub c: Vec<f32>,
    /// Spline degree along x.
    pub kx: usize,
    /// Spline degree along y.
    pub ky: usize,
}

/// `fpbspl`: evaluate the `k+1` non-zero B-splines of degree `k` at the point
/// `x` inside knot interval `[t[l], t[l+1])`, using the stable de Boor–Cox
/// recurrence. Writes the `k+1` values into `h[0..=k]`. `hh` is scratch of
/// length `>= k`. Mirrors `_bispev.pyx:fpbspl` exactly (all f32).
fn fpbspl(t: &[f32], k: usize, x: f32, l: usize, h: &mut [f32], hh: &mut [f32]) {
    h[0] = 1.0;
    for j in 1..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for i in 0..j {
            // `l + i` and `l + i - j` index the knot vector; with l >= k >= j
            // and l <= n-k-1 the indices stay in range, matching the Cython
            // `t[l + i]` / `t[l + i - j]` (l - j >= 0 because l >= k+1 > j here
            // for the interior intervals init_w produces).
            let f = hh[i] / (t[l + i] - t[l + i - j]);
            h[i] += f * (t[l + i] - x);
            h[i + 1] = f * (x - t[l + i - j]);
        }
    }
}

/// `init_w`: for each evaluation point, clamp it to the valid knot span, locate
/// its knot interval, store the base coefficient index `lx[i]` and the `k+1`
/// non-zero basis values `w[i, 0..=k]`. Mirrors `_bispev.pyx:init_w`.
///
/// Returns `(lx, w)` where `w` is flattened row-major as `w[i * (k+1) + j]`.
fn init_w(t: &[f32], k: usize, x: &[f32]) -> (Vec<i32>, Vec<f32>) {
    let n = t.len();
    let m = x.len();
    let k1 = k + 1;
    let mut lx = vec![0i32; m];
    let mut w = vec![0.0f32; m * k1];
    // Scratch buffers, sized as pyFAI does (h: 6, hh: 5) — large enough for the
    // cubic splines this crate handles and any k up to 5.
    let mut h = vec![0.0f32; k1.max(6)];
    let mut hh = vec![0.0f32; k.max(5)];

    let tb = t[k];
    let te = t[n - k - 1];
    let mut l1 = k + 1;
    let mut l2 = l1 + 1;
    for i in 0..m {
        let mut arg = x[i];
        if arg < tb {
            arg = tb;
        }
        if arg > te {
            arg = te;
        }
        // Advance the interval pointer until `arg < t[l1]` or we hit the last
        // usable interval. `l1`/`l2` persist across points (the eval points are
        // monotonic after the caller sorts them, or the loop simply re-scans).
        while !(arg < t[l1] || l1 == (n - k - 1)) {
            l1 = l2;
            l2 = l1 + 1;
        }
        fpbspl(t, k, arg, l1, &mut h, &mut hh);
        lx[i] = (l1 - k - 1) as i32;
        for j in 0..k1 {
            w[i * k1 + j] = h[j];
        }
    }
    (lx, w)
}

/// Evaluate a bivariate B-spline over the cross-product grid of `x` and `y`,
/// returning the surface as a flat row-major `(x.len(), y.len())` array — the
/// transpose pyFAI applies to its internal `(my, mx)` buffer, so element
/// `(ix, iy)` lives at `out[ix * y.len() + iy]`.
///
/// Direct port of `_bispev.pyx:bisplev` + `cy_bispev`: all f32, Kahan summation
/// over the `(kx+1) × (ky+1)` tensor product in `i1`-outer / `j1`-inner order.
pub fn bisplev(x: &[f32], y: &[f32], tck: &Tck) -> Vec<f32> {
    let kx = tck.kx;
    let ky = tck.ky;
    let ny = tck.ty.len();
    let mx = x.len();
    let my = y.len();
    let kx1 = kx + 1;
    let ky1 = ky + 1;
    let nky1 = ny - ky1;

    let (lx, wx) = init_w(&tck.tx, kx, x);
    let (ly, wy) = init_w(&tck.ty, ky, y);

    // pyFAI fills z (shape (my, mx)) then transposes to (mx, my). We build the
    // transposed (mx, my) result directly, preserving the per-point Kahan sum.
    let mut out = vec![0.0f32; mx * my];
    for j in 0..my {
        for i in 0..mx {
            let mut sp = 0.0f32;
            let mut err = 0.0f32;
            for i1 in 0..kx1 {
                for j1 in 0..ky1 {
                    let l2 = (lx[i] as usize) * nky1 + (ly[j] as usize) + i1 * nky1 + j1;
                    let a = tck.c[l2] * wx[i * kx1 + i1] * wy[j * ky1 + j1] - err;
                    let tmp = sp + a;
                    err = (tmp - sp) - a;
                    sp = tmp;
                }
            }
            // z[j*mx + i] += sp (z pre-zeroed) -> transposed to out[i*my + j].
            out[i * my + j] = sp;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A degree-1 (linear) tensor spline reproduces an affine surface exactly
    /// at the knots, independent of pyFAI — a self-contained sanity check that
    /// the recurrence + tensor product are wired correctly.
    #[test]
    fn linear_spline_partition_of_unity() {
        // Clamped linear knots on [0, 2]: [0,0,1,2,2].
        let t = vec![0.0f32, 0.0, 1.0, 2.0, 2.0];
        // 3x3 coefficient grid all ones -> the surface is identically 1.
        let tck = Tck {
            tx: t.clone(),
            ty: t.clone(),
            c: vec![1.0f32; 9],
            kx: 1,
            ky: 1,
        };
        let x = [0.0f32, 0.5, 1.0, 1.5, 2.0];
        let y = [0.0f32, 0.7, 2.0];
        let z = bisplev(&x, &y, &tck);
        for &v in &z {
            assert!((v - 1.0).abs() < 1e-6, "partition of unity, got {v}");
        }
        assert_eq!(z.len(), x.len() * y.len());
    }
}
