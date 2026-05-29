//! Difference-of-Gaussian blob keypoint detection, a port of the deterministic
//! halves of `pyFAI/blob_detection.py` + `pyFAI/ext/_blob.pyx`.
//!
//! The smoothing that builds the DoG scale-space (`gaussian_filter` from
//! `pyFAI/ext/_convolution.pyx`) is not ported here: the DoG stack is taken as
//! input (dumped from pyFAI in the golden). Given that stack, the two operations
//! that locate and refine keypoints are pure comparison / algebra and therefore
//! bit-exact:
//!
//!   * [`local_max`] (`_blob.pyx:49`): a voxel is a keypoint iff it is strictly
//!     greater than all neighbours in the 3x3x3 scale-space cube (and, with
//!     `n_5`, a 5x5 in-plane extension), masked pixels excluded.
//!   * [`refine_Hessian`] (`blob_detection.py:387`): a 3x3x3 second-order
//!     expansion gives the sub-pixel `(x, y, sigma)` offset, the interpolated
//!     peak value, and a validity flag (all offsets `< tresh`).

/// The blob-refinement acceptance threshold (`BlobDetection.tresh`,
/// `blob_detection.py:174`): a refined keypoint is valid iff every sub-pixel
/// offset is below this.
pub const TRESH: f32 = 0.6;

/// A 3-D DoG scale-space, row-major over `(scale, y, x)` of shape
/// `(ns, ny, nx)`, `f32` like pyFAI's `self.dogs`.
pub struct DogStack {
    pub data: Vec<f32>,
    pub ns: usize,
    pub ny: usize,
    pub nx: usize,
}

impl DogStack {
    pub fn new(data: Vec<f32>, ns: usize, ny: usize, nx: usize) -> Self {
        assert_eq!(data.len(), ns * ny * nx, "dog stack length");
        DogStack { data, ns, ny, nx }
    }

    #[inline]
    fn at(&self, s: usize, y: usize, x: usize) -> f32 {
        self.data[(s * self.ny + y) * self.nx + x]
    }
}

/// Keypoint detection (`_blob.local_max`, `_blob.pyx:49`). Returns the
/// `(scale, y, x)` voxel indices of every local maximum, in raster order
/// (scale-major). `mask` (if given) is a row-major `(ny, nx)` invalid-pixel
/// flag; masked positions never produce a keypoint.
pub fn local_max(dogs: &DogStack, mask: Option<&[bool]>, n_5: bool) -> Vec<(usize, usize, usize)> {
    let (ns, ny, nx) = (dogs.ns, dogs.ny, dogs.nx);
    let mut out = Vec::new();
    if ns < 3 || ny < 3 || nx < 3 {
        return out;
    }
    if let Some(m) = mask {
        assert_eq!(m.len(), ny * nx, "mask shape");
    }
    for s in 1..ns - 1 {
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let c = dogs.at(s, y, x);
                if let Some(m) = mask {
                    if m[y * nx + x] {
                        continue;
                    }
                }
                // 3x3x3 strict-max test (verbatim _blob.pyx:81). The duplicated
                // `(c > dogs[s-1,y,x])` term in pyFAI is harmless; we keep the
                // distinct comparisons only.
                let mut m = c > dogs.at(s, y, x - 1)
                    && c > dogs.at(s, y, x + 1)
                    && c > dogs.at(s, y - 1, x)
                    && c > dogs.at(s, y + 1, x)
                    && c > dogs.at(s, y - 1, x - 1)
                    && c > dogs.at(s, y - 1, x + 1)
                    && c > dogs.at(s, y + 1, x - 1)
                    && c > dogs.at(s, y + 1, x + 1)
                    && c > dogs.at(s - 1, y, x)
                    && c > dogs.at(s - 1, y, x - 1)
                    && c > dogs.at(s - 1, y, x + 1)
                    && c > dogs.at(s - 1, y - 1, x)
                    && c > dogs.at(s - 1, y + 1, x)
                    && c > dogs.at(s - 1, y - 1, x - 1)
                    && c > dogs.at(s - 1, y - 1, x + 1)
                    && c > dogs.at(s - 1, y + 1, x - 1)
                    && c > dogs.at(s - 1, y + 1, x + 1)
                    && c > dogs.at(s + 1, y, x - 1)
                    && c > dogs.at(s + 1, y, x + 1)
                    && c > dogs.at(s + 1, y - 1, x)
                    && c > dogs.at(s + 1, y + 1, x)
                    && c > dogs.at(s + 1, y - 1, x - 1)
                    && c > dogs.at(s + 1, y - 1, x + 1)
                    && c > dogs.at(s + 1, y + 1, x - 1)
                    && c > dogs.at(s + 1, y + 1, x + 1);
                if !m {
                    continue;
                }
                if n_5 {
                    if x > 1 {
                        m = m
                            && c > dogs.at(s, y, x - 2)
                            && c > dogs.at(s, y - 1, x - 2)
                            && c > dogs.at(s, y + 1, x - 2)
                            && c > dogs.at(s - 1, y, x - 2)
                            && c > dogs.at(s - 1, y - 1, x - 2)
                            && c > dogs.at(s - 1, y + 1, x - 2)
                            && c > dogs.at(s + 1, y, x - 2)
                            && c > dogs.at(s + 1, y - 1, x - 2)
                            && c > dogs.at(s + 1, y + 1, x - 2);
                        if y > 1 {
                            m = m
                                && c > dogs.at(s, y - 2, x - 2)
                                && c > dogs.at(s - 1, y - 2, x - 2)
                                && c > dogs.at(s, y - 2, x - 2);
                        }
                        if y < ny - 2 {
                            m = m
                                && c > dogs.at(s, y + 2, x - 2)
                                && c > dogs.at(s - 1, y + 2, x - 2)
                                && c > dogs.at(s, y + 2, x - 2);
                        }
                    }
                    if x < nx - 2 {
                        m = m
                            && c > dogs.at(s, y, x + 2)
                            && c > dogs.at(s, y - 1, x + 2)
                            && c > dogs.at(s, y + 1, x + 2)
                            && c > dogs.at(s - 1, y, x + 2)
                            && c > dogs.at(s - 1, y - 1, x + 2)
                            && c > dogs.at(s - 1, y + 1, x + 2)
                            && c > dogs.at(s + 1, y, x + 2)
                            && c > dogs.at(s + 1, y - 1, x + 2)
                            && c > dogs.at(s + 1, y + 1, x + 2);
                        if y > 1 {
                            m = m
                                && c > dogs.at(s, y - 2, x + 2)
                                && c > dogs.at(s - 1, y - 2, x + 2)
                                && c > dogs.at(s, y - 2, x + 2);
                        }
                        if y < ny - 2 {
                            m = m
                                && c > dogs.at(s, y + 2, x + 2)
                                && c > dogs.at(s - 1, y + 2, x + 2)
                                && c > dogs.at(s, y + 2, x + 2);
                        }
                    }
                    if y > 1 {
                        m = m
                            && c > dogs.at(s, y - 2, x)
                            && c > dogs.at(s, y - 2, x - 1)
                            && c > dogs.at(s, y - 2, x + 1)
                            && c > dogs.at(s - 1, y - 2, x)
                            && c > dogs.at(s - 1, y - 2, x - 1)
                            && c > dogs.at(s - 1, y - 2, x + 1)
                            && c > dogs.at(s + 1, y - 2, x)
                            && c > dogs.at(s + 1, y - 2, x - 1)
                            // pyFAI _blob.pyx:117 quirk: this final term reads
                            // dogs[s+1, y+2, x+1] (not y-2). Reproduced verbatim.
                            && c > dogs.at(s + 1, y + 2, x + 1);
                    }
                    if y < ny - 2 {
                        m = m
                            && c > dogs.at(s, y + 2, x)
                            && c > dogs.at(s, y + 2, x - 1)
                            && c > dogs.at(s, y + 2, x + 1)
                            && c > dogs.at(s - 1, y + 2, x)
                            && c > dogs.at(s - 1, y + 2, x - 1)
                            && c > dogs.at(s - 1, y + 2, x + 1)
                            && c > dogs.at(s + 1, y + 2, x)
                            && c > dogs.at(s + 1, y + 2, x - 1)
                            && c > dogs.at(s + 1, y + 2, x + 1);
                    }
                }
                if m {
                    out.push((s, y, x));
                }
            }
        }
    }
    out
}

/// A refined blob keypoint (`refine_Hessian` output, `blob_detection.py:387`).
///
/// pyFAI computes the Hessian deltas in `float32` (numpy `float32` array ops with
/// Python-float literal scalars stay `float32`), but the final positions
/// `kpx + delta_x` promote to `float64` because `kpx` is an `int64` numpy array
/// and `int64 + float32 -> float64`. The peak value `curr + 0.5 * (...)` is a
/// pure `float32` array expression and stays `float32`. We mirror that exactly:
/// positions are `f64`, `peak_val` is `f32`.
#[derive(Debug, Clone, Copy)]
pub struct RefinedKeypoint {
    /// Sub-pixel x position (`kpx + delta_x`, `int64 + float32 -> float64`).
    pub x: f64,
    /// Sub-pixel y position (`kpy + delta_y`, `int64 + float32 -> float64`).
    pub y: f64,
    /// Sub-pixel scale (`kps + delta_s`, `int64 + float32 -> float64`).
    pub sigma: f64,
    /// Interpolated DoG peak value (pure `float32` expression).
    pub peak_val: f32,
    /// Validity: every offset `< TRESH`.
    pub valid: bool,
}

/// Sub-pixel refinement of one keypoint via a 3-point Hessian
/// (`BlobDetection.refine_Hessian`, `blob_detection.py:387`). `(kpx, kpy, kps)`
/// are the integer voxel coordinates from [`local_max`]; the keypoint must be at
/// least one voxel from every border of `dogs`.
pub fn refine_hessian(dogs: &DogStack, kpx: usize, kpy: usize, kps: usize) -> RefinedKeypoint {
    let d = |s: usize, y: usize, x: usize| dogs.at(s, y, x);
    let curr = d(kps, kpy, kpx);
    let nx = d(kps, kpy, kpx + 1);
    let px = d(kps, kpy, kpx - 1);
    let ny = d(kps, kpy + 1, kpx);
    let py = d(kps, kpy - 1, kpx);
    let ns = d(kps + 1, kpy, kpx);
    let ps = d(kps - 1, kpy, kpx);

    let nxny = d(kps, kpy + 1, kpx + 1);
    let nxpy = d(kps, kpy - 1, kpx + 1);
    let pxny = d(kps, kpy + 1, kpx - 1);
    let pxpy = d(kps, kpy - 1, kpx - 1);

    let nsny = d(kps + 1, kpy + 1, kpx);
    let nspy = d(kps + 1, kpy - 1, kpx);
    let psny = d(kps - 1, kpy + 1, kpx);
    let pspy = d(kps - 1, kpy - 1, kpx);

    let nxns = d(kps + 1, kpy, kpx + 1);
    let nxps = d(kps - 1, kpy, kpx + 1);
    let pxns = d(kps + 1, kpy, kpx - 1);
    let pxps = d(kps - 1, kpy, kpx - 1);

    let dx = (nx - px) / 2.0;
    let dy = (ny - py) / 2.0;
    let ds = (ns - ps) / 2.0;
    let dxx = nx - 2.0 * curr + px;
    let dyy = ny - 2.0 * curr + py;
    let dss = ns - 2.0 * curr + ps;
    let dxy = (nxny - nxpy - pxny + pxpy) / 4.0;
    let dxs = (nxns - nxps - pxns + pxps) / 4.0;
    let dsy = (nsny - nspy - psny + pspy) / 4.0;

    let det =
        -(dxs * dyy * dxs) + dsy * dxy * dxs + dxs * dsy * dxy - dss * dxy * dxy - dsy * dsy * dxx
            + dss * dyy * dxx;
    let k00 = dyy * dxx - dxy * dxy;
    let k01 = dxs * dxy - dsy * dxx;
    let k02 = dsy * dxy - dxs * dyy;
    let k10 = dxy * dxs - dsy * dxx;
    let k11 = dss * dxx - dxs * dxs;
    let k12 = dxs * dsy - dss * dxy;
    let k20 = dsy * dxy - dyy * dxs;
    let k21 = dsy * dxs - dss * dxy;
    let k22 = dss * dyy - dsy * dsy;

    let delta_s = -(ds * k00 + dy * k01 + dx * k02) / det;
    let delta_y = -(ds * k10 + dy * k11 + dx * k12) / det;
    let delta_x = -(ds * k20 + dy * k21 + dx * k22) / det;
    let peak_val = curr + 0.5 * (delta_s * ds + delta_y * dy + delta_x * dx);
    let valid = delta_x.abs() < TRESH && delta_y.abs() < TRESH && delta_s.abs() < TRESH;
    // The deltas above are f32 (matching numpy float32 array arithmetic), but the
    // positions promote to f64 because pyFAI adds an int64 keypoint-coordinate
    // array to the float32 delta array (`kpx + delta_x`). Promote each delta to
    // f64 only at the final sum so the f32 rounding of the delta is preserved.
    RefinedKeypoint {
        x: kpx as f64 + delta_x as f64,
        y: kpy as f64 + delta_y as f64,
        sigma: kps as f64 + delta_s as f64,
        peak_val,
        valid,
    }
}
