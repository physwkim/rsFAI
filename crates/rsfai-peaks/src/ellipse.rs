//! Algebraic (non-iterative) ellipse fit, a port of
//! `pyFAI/utils/ellipse.py::fit_ellipse`.
//!
//! The conic is fit by the Fitzgibbon direct method: assemble the design matrix
//! `D = [x², xy, y², x, y, 1]`, the scatter `S = DᵀD`, and solve the generalized
//! eigenproblem `inv(S)·C·v = λ·v` with the constraint matrix `C`. The
//! eigenvector satisfying the three geometric criteria gives the conic
//! coefficients, from which the centre, semi-axes, and angle are recovered.
//!
//! Parity (per `doc/bit-exact-ladder.md`):
//!   * The **design matrix** `D` is per-point products (`x*x`, `x*y`, …) — pure
//!     `f64` `*`, **bit-exact** (Tier A) vs numpy's `numpy.hstack`.
//!   * The scatter `S = DᵀD` is a reduction; numpy routes it through BLAS
//!     (blocked / pairwise summation), so `S` already differs from a sequential
//!     Rust dot by ~1 ULP. The `inv(S)` and the eigensolve are LAPACK
//!     (`dgetri`, `dgeev`) black boxes that `nalgebra` cannot reproduce
//!     bit-for-bit. The recovered ellipse parameters are therefore **Tier-B,
//!     tolerance-gated** at a recorded relative error (≈1e-6, measured in the
//!     golden), **not** a few-ULP budget. This matches the "building blocks
//!     bit-exact; eigensolver-derived output at recorded tolerance" stance.

use nalgebra::DMatrix;

/// A fitted ellipse, mirroring `pyFAI.utils.ellipse.Ellipse`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    /// Centre on the slow axis (y / `pty`).
    pub center_1: f64,
    /// Centre on the fast axis (x / `ptx`).
    pub center_2: f64,
    /// Orientation angle, radians.
    pub angle: f64,
    pub half_long_axis: f64,
    pub half_short_axis: f64,
}

/// Errors raised by [`fit_ellipse`], matching the `ValueError` cases pyFAI
/// raises (`ellipse.py:81,119,140`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EllipseError {
    /// Scatter matrix singular and the delta-shift retry also failed.
    SingularMatrix,
    /// No eigenvector satisfied the three conic criteria.
    NoValidEigenvalue,
    /// `a2 <= 0 || b2 <= 0`: the recovered semi-axes are not real.
    NegativeSqrt,
}

/// Build the design matrix rows `[x², xy, y², x, y, 1]` for each point. This is
/// the bit-exact (Tier-A) building block; exposed so the golden can compare it
/// directly against numpy's `numpy.hstack((x*x, x*y, y*y, x, y, ones))`.
pub fn design_matrix(pty: &[f64], ptx: &[f64]) -> Vec<[f64; 6]> {
    assert_eq!(pty.len(), ptx.len(), "pty/ptx length mismatch");
    pty.iter()
        .zip(ptx.iter())
        .map(|(&y, &x)| [x * x, x * y, y * y, x, y, 1.0])
        .collect()
}

/// Fit an ellipse to the points `(pty, ptx)` (slow, fast coordinates), a port of
/// `fit_ellipse(pty, ptx)`. On a singular scatter matrix the same `+100` shift
/// retry is applied once (`ellipse.py:83`).
pub fn fit_ellipse(pty: &[f64], ptx: &[f64]) -> Result<Ellipse, EllipseError> {
    fit_ellipse_inner(pty, ptx, true)
}

fn fit_ellipse_inner(pty: &[f64], ptx: &[f64], allow_delta: bool) -> Result<Ellipse, EllipseError> {
    let d = design_matrix(pty, ptx);
    let n = d.len();
    // S = Dᵀ D, sequential accumulation (the BLAS-order ULP gap vs numpy is part
    // of the recorded Tier-B tolerance).
    let mut s = DMatrix::<f64>::zeros(6, 6);
    for row in &d {
        for a in 0..6 {
            for b in 0..6 {
                s[(a, b)] += row[a] * row[b];
            }
        }
    }
    let _ = n;

    let inv = match s.try_inverse() {
        Some(inv) => inv,
        None => {
            if !allow_delta {
                return Err(EllipseError::SingularMatrix);
            }
            let delta = 100.0;
            let pty2: Vec<f64> = pty.iter().map(|&v| v + delta).collect();
            let ptx2: Vec<f64> = ptx.iter().map(|&v| v + delta).collect();
            let e = fit_ellipse_inner(&pty2, &ptx2, false)?;
            return Ok(Ellipse {
                center_1: e.center_1 - delta,
                center_2: e.center_2 - delta,
                angle: e.angle,
                half_long_axis: e.half_long_axis,
                half_short_axis: e.half_short_axis,
            });
        }
    };

    // Constraint matrix C (ellipse.py:88).
    let mut c = DMatrix::<f64>::zeros(6, 6);
    c[(0, 2)] = 2.0;
    c[(2, 0)] = 2.0;
    c[(1, 1)] = -1.0;
    let m = &inv * &c;

    // Eigenvalues of M (real ones only). nalgebra returns them via Schur.
    let eigvals = m.clone().complex_eigenvalues();
    // For each finite real eigenvalue, the eigenvector is the right null vector
    // of (M - λI), i.e. the smallest right singular vector. Apply pyFAI's sign
    // and the three criteria (ellipse.py:97-116) and keep the LAST match
    // (numpy's `numpy.where(m)[0][0]` after eigsort-independent masking selects
    // by column order; pyFAI takes the first, but the valid conic eigenvector is
    // unique, so any consistent pick yields the same ellipse).
    let mut chosen: Option<[f64; 6]> = None;
    for k in 0..eigvals.len() {
        let ev = eigvals[k];
        if ev.im != 0.0 || !ev.re.is_finite() {
            continue;
        }
        let mut a_mat = m.clone();
        for i in 0..6 {
            a_mat[(i, i)] -= ev.re;
        }
        let svd = a_mat.svd(true, true);
        let vt = match svd.v_t {
            Some(vt) => vt,
            None => continue,
        };
        // smallest singular value's right vector is the last row of Vᵀ.
        let mut v = [0.0f64; 6];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = vt[(5, i)];
        }
        if v[0] < 0.0 {
            for x in v.iter_mut() {
                *x = -*x;
            }
        }
        let a = v[0];
        let bb = v[1] / 2.0;
        let cc = v[2];
        let dd = v[3] / 2.0;
        let ff = v[4] / 2.0;
        let gg = v[5];
        let delta = a * (cc * gg - ff * ff) - gg * bb * bb + dd * (2.0 * bb * ff - cc * dd);
        let j = a * cc - bb * bb;
        if j > 0.0 && delta != 0.0 && delta * (a + cc) < 0.0 {
            chosen = Some(v);
        }
    }
    let v = chosen.ok_or(EllipseError::NoValidEigenvalue)?;

    let a = v[0];
    let b = v[1] / 2.0;
    let c = v[2];
    let d = v[3] / 2.0;
    let f = v[4] / 2.0;
    let g = v[5];

    let denom = b * b - a * c;
    let x0 = (c * d - b * f) / denom;
    let y0 = (a * f - b * d) / denom;

    let up = 2.0 * (a * f * f + c * d * d + g * b * b - 2.0 * b * d * f - a * c * g);
    let r = (1.0 + 4.0 * b * b / ((a - c) * (a - c))).sqrt();
    let down1 = (b * b - a * c) * ((c - a) * r - (c + a));
    let down2 = (b * b - a * c) * ((a - c) * r - (c + a));
    let a2 = up / down1;
    let b2 = up / down2;
    if a2 <= 0.0 || b2 <= 0.0 {
        return Err(EllipseError::NegativeSqrt);
    }
    let mut res1 = a2.sqrt();
    let mut res2 = b2.sqrt();
    let angle;
    if a == c {
        angle = 0.0;
    } else if res2 > res1 {
        std::mem::swap(&mut res1, &mut res2);
        angle = 0.5 * (std::f64::consts::PI + (2.0 * b).atan2(a - c));
    } else {
        angle = 0.5 * (std::f64::consts::PI + (2.0 * b).atan2(a - c));
    }
    Ok(Ellipse {
        center_1: y0,
        center_2: x0,
        angle,
        half_long_axis: res1,
        half_short_axis: res2,
    })
}
