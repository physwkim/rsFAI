//! Distortion look-up-table build and apply, ported from
//! `pyFAI/ext/_distortion.pyx` (`calc_pos`, `calc_sparse`, `correct`) and the
//! pixel-polygon clipper `_integrate2d` from `pyFAI/ext/regrid_common.pxi`.
//!
//! The distortion correction maps each *raw* detector pixel — a quadrilateral
//! whose four corners land at sub-pixel positions on a regular output grid — to
//! the output bins it overlaps, weighting each by the fractional area. The
//! per-pixel clipping (`_integrate2d` of the four polygon edges into a small
//! `box` buffer) and the area normalisation are done in f32, exactly as the
//! Cython `buffer_t`/`float` code; the CSR build records `(pixel idx, bin,
//! coefficient)` triples in raster order, then scatters them stably into the
//! CSR so each bin's entries stay in ascending pixel-index order.
//!
//! Apply (`correct`) sums `lin[idx] * coef` per output bin in a **double**
//! accumulator (the `correct_CSR_double` path), then stamps empty bins with the
//! `dummy` value (default 0).

use rsfai_core::dtype::{BufferT, DataT, IndexT};

/// Result of [`calc_pos`]: the per-corner positions on the output grid plus the
/// derived grid metadata, mirroring `_distortion.calc_pos`'s 5-tuple.
#[derive(Debug, Clone)]
pub struct CalcPos {
    /// Pixel-corner positions, flat layout `pos[((i*ncol + j)*4 + k)*2 + d]`
    /// for pixel `(i, j)`, corner `k` in `0..4`, dim `d` in `{0, 1}`. f32.
    pub pos: Vec<f32>,
    /// Number of detector rows (dim0) and columns (dim1).
    pub shape_in: (usize, usize),
    /// Max pixel extent along each axis (`delta0`, `delta1`) — the buffer size.
    pub delta: (usize, usize),
    /// Output image shape `(shape_out0, shape_out1)`.
    pub shape_out: (usize, usize),
    /// Position of the first bin `(offset0, offset1)` (zero unless resizing).
    pub offset: (f64, f64),
}

/// A distortion look-up table in CSR (compressed-sparse-row) form, the
/// `(data, indices, indptr)` triple `calc_sparse(format="csr")` returns.
#[derive(Debug, Clone)]
pub struct Csr {
    /// Fractional-area coefficients, one per non-zero LUT entry. f32.
    pub data: Vec<DataT>,
    /// Raw-pixel index for each coefficient. i32.
    pub indices: Vec<IndexT>,
    /// Row pointers, length `shape_out0*shape_out1 + 1`. i32.
    pub indptr: Vec<IndexT>,
}

/// `floor(min(a, b, c, d))` in f32 (`_distortion.pyx:_floor_min4`).
#[inline]
fn floor_min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let mut res = if b < a { b } else { a };
    if c < res {
        res = c;
    }
    if d < res {
        res = d;
    }
    res.floor()
}

/// `ceil(max(a, b, c, d))` in f32 (`_distortion.pyx:_ceil_max4`).
#[inline]
fn ceil_max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let mut res = if b > a { b } else { a };
    if c > res {
        res = c;
    }
    if d > res {
        res = d;
    }
    res.ceil()
}

/// `_calc_area(I1, I2, slope, intercept)` from `regrid_common.pxi`:
/// `(I2 - I1) * (0.5 * slope * (I2 + I1) + intercept)`.
///
/// Called inside `_integrate2d` with `floating = float`, so all four inputs are
/// `float`. The C body follows the usual-arithmetic-conversion rules per
/// sub-expression, NOT one global widening:
///
/// - `(I2 - I1)` and `(I2 + I1)` are `float - float` / `float + float`, each
///   rounded to **f32** before being used further.
/// - `0.5 * slope` introduces the `double` literal `0.5`, so from there the
///   inner parenthesis `0.5 * slope * (I2 + I1) + intercept` is evaluated in
///   **double** (the f32 `(I2 + I1)` and `intercept` promote).
/// - the outer `(I2 - I1) * (...)` is then `float * double` → **double**, and
///   the result casts back to `float` on return.
///
/// So `(I2 - I1)` / `(I2 + I1)` carry an f32 rounding step; only the products
/// and the final sum run in double. Casting `I1`/`I2` to f64 up front (computing
/// the differences in f64) skips that rounding and diverges by a few ULP.
#[inline]
fn calc_area(i1: f32, i2: f32, slope: f32, intercept: f32) -> f32 {
    let diff = (i2 - i1) as f64;
    let sum = (i2 + i1) as f64;
    let slope = slope as f64;
    let intercept = intercept as f64;
    (diff * (0.5 * slope * sum + intercept)) as f32
}

/// Calculate the pixel-corner positions on the regular output grid, the port of
/// `_distortion.calc_pos`. `pixel_corners` is the detector's `(nrow, ncol, 4,
/// 3)` corner array flattened row-major (vertex order z, y, x); we use the y
/// (index 1) and x (index 2) components divided by `pixel1`/`pixel2`.
///
/// `shape_out`, when `Some`, fixes the output shape and zeroes the offset
/// (pyFAI's `resize=False`); when `None`, the shape and offset are derived from
/// the corner bounding box.
pub fn calc_pos(
    pixel_corners: &[f32],
    shape_in: (usize, usize),
    pixel1: f64,
    pixel2: f64,
    shape_out: Option<(usize, usize)>,
) -> CalcPos {
    assert!(
        pixel1 != 0.0 && pixel2 != 0.0,
        "pixel size cannot be null (division by zero)"
    );
    let (dim0, dim1) = shape_in;
    let p1 = pixel1 as f32;
    let p2 = pixel2 as f32;
    let do_shape = shape_out.is_none();
    let big = f32::MAX;

    let mut pos = vec![0.0f32; dim0 * dim1 * 4 * 2];
    let mut delta0 = -big;
    let mut delta1 = -big;
    let mut all_min0 = big;
    let mut all_min1 = big;
    let mut all_max0 = -big;
    let mut all_max1 = -big;

    for i in 0..dim0 {
        for j in 0..dim1 {
            let mut min0 = big;
            let mut min1 = big;
            let mut max0 = -big;
            let mut max1 = -big;
            for k in 0..4 {
                // pixel_corners[i, j, k, 1] (y) and [i, j, k, 2] (x).
                let base = ((i * dim1 + j) * 4 + k) * 3;
                let v0 = pixel_corners[base + 1] / p1;
                let v1 = pixel_corners[base + 2] / p2;
                let pbase = ((i * dim1 + j) * 4 + k) * 2;
                pos[pbase] = v0;
                pos[pbase + 1] = v1;
                if v0 < min0 {
                    min0 = v0;
                }
                if v1 < min1 {
                    min1 = v1;
                }
                if v0 > max0 {
                    max0 = v0;
                }
                if v1 > max1 {
                    max1 = v1;
                }
            }
            delta0 = delta0.max(max0.ceil() - min0.floor());
            delta1 = delta1.max(max1.ceil() - min1.floor());
            if do_shape {
                if min0 < all_min0 {
                    all_min0 = min0;
                }
                if min1 < all_min1 {
                    all_min1 = min1;
                }
                if max0 > all_max0 {
                    all_max0 = max0;
                }
                if max1 > all_max1 {
                    all_max1 = max1;
                }
            }
        }
    }

    let (shape_out, offset) = if do_shape {
        (
            (
                (all_max0 - all_min0).ceil() as usize,
                (all_max1 - all_min1).ceil() as usize,
            ),
            (all_min0 as f64, all_min1 as f64),
        )
    } else {
        (shape_out.unwrap(), (0.0, 0.0))
    };

    CalcPos {
        pos,
        shape_in,
        delta: (delta0 as i32 as usize, delta1 as i32 as usize),
        shape_out,
        offset,
    }
}

/// Integrate one polygon edge `start -> stop` (coordinates relative to the
/// pixel's bounding box) into the area `box`, the port of
/// `regrid_common.pxi:_integrate2d`. `box` is a flat `(rows, cols)` buffer of
/// width `cols`; the first coordinate selects the row, `h` walks the columns.
/// All arithmetic is f32, with `copysign(dA, segment_area)` as in the Cython.
#[allow(clippy::too_many_arguments)]
fn integrate2d(
    boxbuf: &mut [BufferT],
    cols: usize,
    start0: f32,
    start1: f32,
    stop0: f32,
    stop1: f32,
) {
    if start0 == stop0 {
        return;
    }
    let slope = (stop1 - start1) / (stop0 - start0);
    let intercept = stop1 - slope * stop0;

    // box[r, h] += copysign(dA, segment_area). In the Cython, `copysign` is the
    // C `double copysign(double, double)` and the `float +=` promotes the cell
    // to double, adds, and rounds back to f32 — so the per-write add is done in
    // f64. `val` here is the already-signed contribution in f64; we add it to
    // the f64-widened cell and round once.
    let mut put = |r: i64, h: usize, val: f64| {
        if r < 0 {
            return;
        }
        let r = r as usize;
        let idx = r * cols + h;
        if idx < boxbuf.len() {
            boxbuf[idx] = (boxbuf[idx] as f64 + val) as f32;
        }
    };

    if start0 < stop0 {
        // positive contribution
        let mut p = start0.ceil();
        let dp = p - start0;
        if p > stop0 {
            // start0 and stop0 are in the same unit cell
            let segment_area = calc_area(start0, stop0, slope, intercept);
            if segment_area != 0.0 {
                let mut abs_area = segment_area.abs();
                let mut da = stop0 - start0;
                let mut h = 0usize;
                while abs_area > 0.0 && h < cols {
                    if da > abs_area {
                        da = abs_area;
                        abs_area = -1.0;
                    }
                    put(start0 as i64, h, (da as f64).copysign(segment_area as f64));
                    abs_area -= da;
                    h += 1;
                }
            }
        } else {
            if dp > 0.0 {
                let segment_area = calc_area(start0, p, slope, intercept);
                if segment_area != 0.0 {
                    let mut abs_area = segment_area.abs();
                    let mut h = 0usize;
                    let mut da = dp;
                    while abs_area > 0.0 && h < cols {
                        if da > abs_area {
                            da = abs_area;
                            abs_area = -1.0;
                        }
                        put(p as i64 - 1, h, (da as f64).copysign(segment_area as f64));
                        abs_area -= da;
                        h += 1;
                    }
                }
            }
            // subsection P1 -> Pn
            let from = p.floor() as i64;
            let to = stop0.floor() as i64;
            let mut i = from;
            while i < to {
                let segment_area = calc_area(i as f32, (i + 1) as f32, slope, intercept);
                if segment_area != 0.0 {
                    let mut abs_area = segment_area.abs();
                    let mut h = 0usize;
                    let mut da = 1.0f32;
                    while abs_area > 0.0 && h < cols {
                        if da > abs_area {
                            da = abs_area;
                            abs_area = -1.0;
                        }
                        put(i, h, (da as f64).copysign(segment_area as f64));
                        abs_area -= da;
                        h += 1;
                    }
                }
                i += 1;
            }
            // Section Pn -> B
            p = stop0.floor();
            let dp2 = stop0 - p;
            if dp2 > 0.0 {
                let segment_area = calc_area(p, stop0, slope, intercept);
                if segment_area != 0.0 {
                    let mut abs_area = segment_area.abs();
                    let mut h = 0usize;
                    let mut da = dp2.abs();
                    while abs_area > 0.0 && h < cols {
                        if da > abs_area {
                            da = abs_area;
                            abs_area = -1.0;
                        }
                        put(p as i64, h, (da as f64).copysign(segment_area as f64));
                        abs_area -= da;
                        h += 1;
                    }
                }
            }
        }
    } else {
        // start0 > stop0: negative contribution
        let p = start0.floor();
        if stop0 > p {
            // start0 and stop0 are in the same unit cell
            let segment_area = calc_area(start0, stop0, slope, intercept);
            if segment_area != 0.0 {
                let mut abs_area = segment_area.abs();
                let mut da = start0 - stop0;
                let mut h = 0usize;
                while abs_area > 0.0 && h < cols {
                    if da > abs_area {
                        da = abs_area;
                        abs_area = -1.0;
                    }
                    put(start0 as i64, h, (da as f64).copysign(segment_area as f64));
                    abs_area -= da;
                    h += 1;
                }
            }
        } else {
            let dp = p - start0;
            if dp < 0.0 {
                let segment_area = calc_area(start0, p, slope, intercept);
                if segment_area != 0.0 {
                    let mut abs_area = segment_area.abs();
                    let mut h = 0usize;
                    let mut da = dp.abs();
                    while abs_area > 0.0 && h < cols {
                        if da > abs_area {
                            da = abs_area;
                            abs_area = -1.0;
                        }
                        put(p as i64, h, (da as f64).copysign(segment_area as f64));
                        abs_area -= da;
                        h += 1;
                    }
                }
            }
            // subsection P1 -> Pn (descending)
            let from = start0 as i64;
            let to = stop0.ceil() as i64;
            let mut i = from;
            while i > to {
                let segment_area = calc_area(i as f32, (i - 1) as f32, slope, intercept);
                if segment_area != 0.0 {
                    let mut abs_area = segment_area.abs();
                    let mut h = 0usize;
                    let mut da = 1.0f32;
                    while abs_area > 0.0 && h < cols {
                        if da > abs_area {
                            da = abs_area;
                            abs_area = -1.0;
                        }
                        put(i - 1, h, (da as f64).copysign(segment_area as f64));
                        abs_area -= da;
                        h += 1;
                    }
                }
                i -= 1;
            }
            // Section Pn -> B
            let p2 = stop0.ceil();
            let dp2 = stop0 - p2;
            if dp2 < 0.0 {
                let segment_area = calc_area(p2, stop0, slope, intercept);
                if segment_area != 0.0 {
                    let mut abs_area = segment_area.abs();
                    let mut h = 0usize;
                    let mut da = dp2.abs();
                    while abs_area > 0.0 && h < cols {
                        if da > abs_area {
                            da = abs_area;
                            abs_area = -1.0;
                        }
                        put(stop0 as i64, h, (da as f64).copysign(segment_area as f64));
                        abs_area -= da;
                        h += 1;
                    }
                }
            }
        }
    }
}

/// Build the distortion CSR look-up table, the port of
/// `_distortion.calc_sparse(format="csr")`. `cp` carries the corner positions
/// and grid metadata from [`calc_pos`]; `mask` (when present, length
/// `shape_in0*shape_in1`, non-zero = masked) skips raw pixels; `offset` is the
/// global pixel offset (`(0, 0)` unless the detector is resized).
///
/// The output bins are ordered raster-major (`bin = ml*shape_out1 + nl`); each
/// bin's entries are in ascending raw-pixel-index order, matching pyFAI's
/// stable scatter.
pub fn calc_sparse(cp: &CalcPos, mask: Option<&[i8]>, offset: (f64, f64)) -> Csr {
    let (shape_in0, shape_in1) = cp.shape_in;
    let (shape_out0, shape_out1) = cp.shape_out;
    let (mut delta0, mut delta1) = cp.delta;
    let bins = shape_out0 * shape_out1;
    let size_in = shape_in0 * shape_in1;
    let goffset0 = offset.0 as f32;
    let goffset1 = offset.1 as f32;

    // Triples accumulated in raster (pixel) order: (pixel idx, bin, value).
    let mut idx_pixel: Vec<IndexT> = Vec::new();
    let mut idx_bin: Vec<IndexT> = Vec::new();
    let mut large_data: Vec<DataT> = Vec::new();
    let mut pixel_count = vec![0i32; bins];

    let mut buffer = vec![0.0 as BufferT; delta0 * delta1];

    for idx in 0..size_in {
        let i = idx / shape_in1;
        let j = idx % shape_in1;
        if let Some(m) = mask {
            if m[i * shape_in1 + j] != 0 {
                continue;
            }
        }
        // Reset the area buffer.
        for b in buffer.iter_mut() {
            *b = 0.0;
        }

        let pb = |k: usize, d: usize| -> f32 { cp.pos[((i * shape_in1 + j) * 4 + k) * 2 + d] };
        let mut a0 = pb(0, 0) - goffset0;
        let mut a1 = pb(0, 1) - goffset1;
        let mut b0 = pb(1, 0) - goffset0;
        let mut b1 = pb(1, 1) - goffset1;
        let mut c0 = pb(2, 0) - goffset0;
        let mut c1 = pb(2, 1) - goffset1;
        let mut d0 = pb(3, 0) - goffset0;
        let mut d1 = pb(3, 1) - goffset1;

        let foffset0 = floor_min4(a0, b0, c0, d0);
        let foffset1 = floor_min4(a1, b1, c1, d1);
        let offset0 = foffset0 as i32;
        let offset1 = foffset1 as i32;
        let box_size0 = ceil_max4(a0, b0, c0, d0) as i32 - offset0;
        let box_size1 = ceil_max4(a1, b1, c1, d1) as i32 - offset1;
        if box_size0 as usize > delta0 || box_size1 as usize > delta1 {
            // Grow the buffer (pyFAI uses max(offset, delta) — the offset, not
            // the box size; we mirror that, then re-zero).
            delta0 = (offset0 as usize).max(delta0);
            delta1 = (offset1 as usize).max(delta1);
            buffer = vec![0.0 as BufferT; delta0 * delta1];
        }

        a0 -= foffset0;
        a1 -= foffset1;
        b0 -= foffset0;
        b1 -= foffset1;
        c0 -= foffset0;
        c1 -= foffset1;
        d0 -= foffset0;
        d1 -= foffset1;

        // ABCD in trigonometric order: edges B->A, C->B, D->C, A->D.
        integrate2d(&mut buffer, delta1, b0, b1, a0, a1);
        integrate2d(&mut buffer, delta1, c0, c1, b0, b1);
        integrate2d(&mut buffer, delta1, d0, d1, c0, c1);
        integrate2d(&mut buffer, delta1, a0, a1, d0, d1);

        let area = 0.5 * ((c0 - a0) * (d1 - b1) - (c1 - a1) * (d0 - b0));
        let inv_area = 1.0 / area;
        for ms in 0..box_size0 {
            let ml = ms + offset0;
            if ml < 0 || ml >= shape_out0 as i32 {
                continue;
            }
            for ns in 0..box_size1 {
                let nl = ns + offset1;
                if nl < 0 || nl >= shape_out1 as i32 {
                    continue;
                }
                let value = buffer[ms as usize * delta1 + ns as usize] * inv_area;
                if value == 0.0 {
                    continue;
                }
                if !(0.0..=1.0001).contains(&value) {
                    // Pathological clip result — pyFAI logs and skips it.
                    continue;
                }
                let bin_number = ml * shape_out1 as i32 + nl;
                pixel_count[bin_number as usize] += 1;
                idx_pixel.push(idx as IndexT);
                idx_bin.push(bin_number);
                large_data.push(value);
            }
        }
    }

    // CSR assembly: cumulative row pointers, then a stable scatter so that each
    // bin's entries remain in the raster (ascending pixel-index) order.
    let mut indptr = vec![0i32; bins + 1];
    let mut acc = 0i32;
    for i in 0..bins {
        indptr[i] = acc;
        acc += pixel_count[i];
    }
    indptr[bins] = acc;
    let lut_size = acc as usize;

    let mut indices = vec![0i32; lut_size];
    let mut data = vec![0.0 as DataT; lut_size];
    let mut fill = vec![0i32; bins];
    for n in 0..idx_bin.len() {
        let bin_number = idx_bin[n] as usize;
        let i = indptr[bin_number] + fill[bin_number];
        fill[bin_number] += 1;
        indices[i as usize] = idx_pixel[n];
        data[i as usize] = large_data[n];
    }

    Csr {
        data,
        indices,
        indptr,
    }
}

/// Apply a distortion CSR LUT to an image, the port of
/// `_distortion.correct` -> `correct_CSR_double`. `image` is the flat raw image
/// (length `shape_in0*shape_in1`); the output is the flat corrected image
/// (length `shape_out0*shape_out1`).
///
/// Each output bin sums `image[idx] * coef` over its CSR row in a **double**
/// accumulator; an empty bin (no contributing pixel, `sum == 0`) is stamped
/// with `dummy` (pyFAI's `self.empty`, default 0).
pub fn correct(image: &[DataT], lut: &Csr, dummy: f32) -> Vec<DataT> {
    let bins = lut.indptr.len() - 1;
    let size = image.len() as i32;
    let mut out = vec![0.0 as DataT; bins];

    for (i, slot) in out.iter_mut().enumerate() {
        let mut sum: f64 = 0.0;
        let lo = lut.indptr[i] as usize;
        let hi = lut.indptr[i + 1] as usize;
        for j in lo..hi {
            let idx = lut.indices[j];
            let coef = lut.data[j];
            if coef <= 0.0 {
                continue;
            }
            if idx >= size {
                continue;
            }
            let value = image[idx as usize];
            sum += value as f64 * coef as f64;
        }
        if sum == 0.0 {
            *slot = dummy;
        } else {
            *slot = sum as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit-square pixel exactly aligned to the output grid maps entirely to
    /// one bin with coefficient 1 — a self-contained check of the clip + area
    /// normalisation independent of pyFAI.
    #[test]
    fn axis_aligned_unit_pixel_maps_to_one_bin() {
        // Pixel (0,0) corners ABCD = (1,1),(2,1),(2,2),(1,2) in grid units:
        // a 1x1 square covering output bin (1,1).
        // pixel_corners layout [.. , k, (z,y,x)]; set y/x, z=0.
        let corners = vec![
            // i=0,j=0
            0.0, 1.0, 1.0, // A
            0.0, 2.0, 1.0, // B
            0.0, 2.0, 2.0, // C
            0.0, 1.0, 2.0, // D
        ];
        let cp = calc_pos(&corners, (1, 1), 1.0, 1.0, Some((4, 4)));
        let csr = calc_sparse(&cp, None, (0.0, 0.0));
        // Bin (1,1) = index 1*4 + 1 = 5 should hold the whole pixel (coef ~1).
        let row_lo = csr.indptr[5] as usize;
        let row_hi = csr.indptr[6] as usize;
        assert_eq!(row_hi - row_lo, 1, "exactly one contributor to bin 5");
        assert_eq!(csr.indices[row_lo], 0, "raw pixel 0");
        assert!(
            (csr.data[row_lo] - 1.0).abs() < 1e-5,
            "coef ~ 1, got {}",
            csr.data[row_lo]
        );

        // correct: an image with value 7 at pixel 0 -> bin 5 holds 7.
        let out = correct(&[7.0], &csr, 0.0);
        assert!((out[5] - 7.0).abs() < 1e-4, "bin 5 = 7, got {}", out[5]);
        assert_eq!(out[0], 0.0, "empty bins stay at dummy 0");
    }
}
