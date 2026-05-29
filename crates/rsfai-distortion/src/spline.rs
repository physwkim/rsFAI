//! FITPACK `.spline` ASCII file parser and displacement-map evaluation, ported
//! from `pyFAI/spline.py` (`Spline.read` / `Spline.spline2array`).
//!
//! A `.spline` file (Fit2D / SPD format) describes the spatial distortion of a
//! 2D detector as two bicubic B-spline surfaces — one for the X displacement,
//! one for Y. The file stores, for each, the knot vectors and coefficient grid;
//! evaluating them over the pixel grid with [`bisplev`](crate::bispev) yields
//! the per-pixel displacement in pixel units.
//!
//! Floats are stored as fixed-width 14-character fields (`lenStrFloat = 14`),
//! with the sign packed against the mantissa (e.g. `-0.2227855E+02`), so the
//! parser slices the line into 14-char chunks rather than splitting on spaces —
//! matching `Spline.read` verbatim.

use std::path::Path;

use crate::bispev::{bisplev, Tck};
use crate::error::{DistortionError, Result};

/// Cubic spline order (`Spline.splineOrder = 3`).
const SPLINE_ORDER: usize = 3;
/// Fixed-width float field length (`Spline.lenStrFloat = 14`).
const LEN_STR_FLOAT: usize = 14;

/// A parsed `.spline` distortion file (`pyFAI.spline.Spline`).
///
/// Holds the valid region, grid/pixel metadata, and the two bicubic B-spline
/// tensors (X- and Y-displacement). Knots and coefficients are f32, matching
/// the `numpy.float32` arrays `Spline.read` builds.
#[derive(Debug, Clone)]
pub struct Spline {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
    /// Grid spacing (microns) from the `GRID SPACING` line.
    pub grid: f64,
    /// `(x_pixel_size, y_pixel_size)` in microns.
    pub pixel_size: (f64, f64),
    /// Cubic spline degree (always 3 for these files).
    pub order: usize,
    /// X-displacement tensor: knots `tx`/`ty`, coefficients `c`, degree 3.
    pub x_tck: Tck,
    /// Y-displacement tensor.
    pub y_tck: Tck,
}

/// Parse the leading `count` fixed-width 14-char float fields out of `line`,
/// matching the slice-based parse in `Spline.read`. A short final field (the
/// line ends mid-field) is taken as the remaining characters.
fn parse_fixed_floats(line: &str, count: usize, out: &mut Vec<f64>) -> Result<()> {
    let bytes = line.as_bytes();
    for i in 0..count {
        let start = i * LEN_STR_FLOAT;
        if start >= bytes.len() {
            break;
        }
        let end = ((i + 1) * LEN_STR_FLOAT).min(bytes.len());
        let field = line[start..end].trim();
        if field.is_empty() {
            continue;
        }
        let v: f64 = field
            .parse()
            .map_err(|_| DistortionError::Parse(format!("spline: bad float field {field:?}")))?;
        out.push(v);
    }
    Ok(())
}

/// Parse every 14-char float field on a line (used for the distortion data
/// blocks, where each line holds `len(line) // 14` floats).
fn parse_all_floats(line: &str, out: &mut Vec<f64>) -> Result<()> {
    let n = line.len() / LEN_STR_FLOAT;
    parse_fixed_floats(line, n, out)
}

impl Spline {
    /// Read and parse a `.spline` file from disk (`Spline.read`).
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Spline> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p).map_err(|source| DistortionError::Io {
            path: p.to_string_lossy().into_owned(),
            source,
        })?;
        Spline::parse(&text)
    }

    /// Parse the contents of a `.spline` file (the body of [`Spline::read`],
    /// factored out so tests can feed a string).
    pub fn parse(text: &str) -> Result<Spline> {
        // pyFAI rstrips each line (keeps leading spaces, which carry field
        // alignment); it does NOT strip trailing whitespace differences that
        // would shift the 14-char windows.
        let lines: Vec<&str> = text.lines().map(|l| l.trim_end()).collect();

        let mut xmin = None;
        let mut ymin = None;
        let mut xmax = None;
        let mut ymax = None;
        let mut grid = None;
        let mut pixel_size = None;
        let mut x_knots_x = Vec::new();
        let mut x_knots_y = Vec::new();
        let mut x_coeff = Vec::new();
        let mut y_knots_x = Vec::new();
        let mut y_knots_y = Vec::new();
        let mut y_coeff = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            let tag = line.trim().to_ascii_uppercase();
            match tag.as_str() {
                "VALID REGION" => {
                    let data = lines.get(idx + 1).copied().unwrap_or("");
                    let mut v = Vec::new();
                    parse_fixed_floats(data, 4, &mut v)?;
                    if v.len() < 4 {
                        return Err(DistortionError::Parse(
                            "spline: VALID REGION needs 4 floats".into(),
                        ));
                    }
                    xmin = Some(v[0]);
                    ymin = Some(v[1]);
                    xmax = Some(v[2]);
                    ymax = Some(v[3]);
                }
                "GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE" => {
                    let data = lines.get(idx + 1).copied().unwrap_or("");
                    let mut v = Vec::new();
                    parse_fixed_floats(data, 3, &mut v)?;
                    if v.len() < 3 {
                        return Err(DistortionError::Parse(
                            "spline: GRID SPACING needs 3 floats".into(),
                        ));
                    }
                    grid = Some(v[0]);
                    pixel_size = Some((v[1], v[2]));
                }
                "X-DISTORTION" => {
                    let (kx, ky, coeff) = parse_distortion_block(&lines, idx)?;
                    x_knots_x = kx;
                    x_knots_y = ky;
                    x_coeff = coeff;
                }
                "Y-DISTORTION" => {
                    let (kx, ky, coeff) = parse_distortion_block(&lines, idx)?;
                    y_knots_x = kx;
                    y_knots_y = ky;
                    y_coeff = coeff;
                }
                _ => {}
            }
        }

        let xmin =
            xmin.ok_or_else(|| DistortionError::Parse("spline: missing VALID REGION".into()))?;
        let ymin = ymin.unwrap();
        let xmax = xmax.unwrap();
        let ymax = ymax.unwrap();
        let grid =
            grid.ok_or_else(|| DistortionError::Parse("spline: missing GRID SPACING".into()))?;
        let pixel_size = pixel_size.unwrap();

        if x_coeff.is_empty() || y_coeff.is_empty() {
            return Err(DistortionError::Parse(
                "spline: missing X-/Y-DISTORTION coefficients".into(),
            ));
        }

        Ok(Spline {
            xmin,
            ymin,
            xmax,
            ymax,
            grid,
            pixel_size,
            order: SPLINE_ORDER,
            x_tck: Tck {
                tx: x_knots_x,
                ty: x_knots_y,
                c: x_coeff,
                kx: SPLINE_ORDER,
                ky: SPLINE_ORDER,
            },
            y_tck: Tck {
                tx: y_knots_x,
                ty: y_knots_y,
                c: y_coeff,
                kx: SPLINE_ORDER,
                ky: SPLINE_ORDER,
            },
        })
    }

    /// Evaluate the X- and Y-displacement maps over the detector grid, the port
    /// of `Spline.spline2array`: `x = arange(xmin, xmax+1)`,
    /// `y = arange(ymin, ymax+1)`, then `bisplev(x, y, tck).T`.
    ///
    /// Returns `(x_disp, y_disp)`, each a flat row-major `(ny, nx)` array where
    /// `ny = len(y)` and `nx = len(x)` — the shape after pyFAI's `.transpose()`.
    pub fn spline2array(&self) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = arange_inclusive_plus_one(self.xmin, self.xmax);
        let y: Vec<f32> = arange_inclusive_plus_one(self.ymin, self.ymax);
        let x_disp = transpose(&bisplev(&x, &y, &self.x_tck), x.len(), y.len());
        let y_disp = transpose(&bisplev(&x, &y, &self.y_tck), x.len(), y.len());
        (x_disp, y_disp)
    }

    /// The detector grid axes `spline2array` evaluates over (`arange(min, max+1)`
    /// as f32), exposed so callers can drive [`bisplev`](crate::bispev::bisplev)
    /// or inspect the grid without recomputing.
    pub fn grid_axes(&self) -> (Vec<f32>, Vec<f32>) {
        (
            arange_inclusive_plus_one(self.xmin, self.xmax),
            arange_inclusive_plus_one(self.ymin, self.ymax),
        )
    }
}

/// Parse one `X-DISTORTION`/`Y-DISTORTION` block starting at the header line
/// index `idx`: the next line holds `(nKnotsX, nKnotsY)` as two ints, then the
/// float block runs line-by-line until a blank line. Returns `(knotsX, knotsY,
/// coeff)`, all f32, split by the two knot counts.
fn parse_distortion_block(lines: &[&str], idx: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let header = lines
        .get(idx + 1)
        .copied()
        .ok_or_else(|| DistortionError::Parse("spline: distortion header missing".into()))?;
    let counts: Vec<usize> = header
        .split_whitespace()
        .map(|t| t.parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| DistortionError::Parse(format!("spline: bad knot-count line {header:?}")))?;
    if counts.len() < 2 {
        return Err(DistortionError::Parse(
            "spline: distortion header needs two ints".into(),
        ));
    }
    let (n_kx, n_ky) = (counts[0], counts[1]);

    let mut datablock = Vec::new();
    for line in &lines[idx + 2..] {
        if line.is_empty() {
            break;
        }
        parse_all_floats(line, &mut datablock)?;
    }
    if datablock.len() < n_kx + n_ky {
        return Err(DistortionError::Parse(
            "spline: distortion block shorter than knot counts".into(),
        ));
    }
    let knots_x: Vec<f32> = datablock[..n_kx].iter().map(|&v| v as f32).collect();
    let knots_y: Vec<f32> = datablock[n_kx..n_kx + n_ky]
        .iter()
        .map(|&v| v as f32)
        .collect();
    let coeff: Vec<f32> = datablock[n_kx + n_ky..].iter().map(|&v| v as f32).collect();
    Ok((knots_x, knots_y, coeff))
}

/// `numpy.arange(start, stop + 1)` cast to f32 — the per-axis grid
/// `spline2array` uses. numpy's `arange` with the default step 1 yields
/// `[start, start+1, ..., <= stop]`; with integer-valued bounds this is exactly
/// `stop - start + 1` points.
fn arange_inclusive_plus_one(start: f64, stop: f64) -> Vec<f32> {
    // numpy.arange(start, stop+1) length = ceil((stop+1 - start)). For the
    // integer-valued region bounds in a spline file this is stop-start+1.
    let n = ((stop + 1.0) - start).ceil() as usize;
    (0..n).map(|i| (start + i as f64) as f32).collect()
}

/// Transpose a flat row-major `(rows, cols)` array to `(cols, rows)`. Used to
/// turn `bisplev`'s `(nx, ny)` result into the `(ny, nx)` map `spline2array`
/// returns (pyFAI's `.transpose()`).
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALFCCD: &str = include_str!("../../../golden/datasets_distortion/halfccd.spline");

    #[test]
    fn parse_halfccd_header() {
        let sp = Spline::parse(HALFCCD).expect("parse");
        assert_eq!(sp.xmin, 0.0);
        assert_eq!(sp.ymin, 0.0);
        assert_eq!(sp.xmax, 2048.0);
        assert_eq!(sp.ymax, 1025.0);
        assert_eq!(sp.order, 3);
        // 8 knots each direction, 16 coefficients each (cubic 4x4 grid).
        assert_eq!(sp.x_tck.tx.len(), 8);
        assert_eq!(sp.x_tck.ty.len(), 8);
        assert_eq!(sp.x_tck.c.len(), 16);
        assert_eq!(sp.y_tck.tx.len(), 8);
        assert_eq!(sp.y_tck.c.len(), 16);
    }

    #[test]
    fn fixed_width_handles_packed_sign() {
        // A line where a negative value abuts the previous field with no space.
        let line = "-0.2227855E+02-0.2393550E+02";
        let mut v = Vec::new();
        parse_fixed_floats(line, 2, &mut v).unwrap();
        assert_eq!(v.len(), 2);
        assert!((v[0] - (-22.27855)).abs() < 1e-4);
        assert!((v[1] - (-23.93550)).abs() < 1e-4);
    }
}
