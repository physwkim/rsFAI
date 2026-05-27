//! The detector model, ported from `pyFAI/detectors/_common.py` (base
//! `Detector` + `ModuleDetector`) and `_dectris.py` (Pilatus/Eiger).
//!
//! Scope (M2): the flat, contiguous module detectors the golden datasets need —
//! Pilatus1M and the Eiger family — plus a generic `(pixel1, pixel2, shape)`
//! detector. These differ only in pixel size, module layout, and the static
//! gap mask; the pixel-centre recipe is identical:
//!
//! ```text
//! p1 = pixel1 * (d1 + 0.5)   p2 = pixel2 * (d2 + 0.5)   p3 = None (flat)
//! ```
//!
//! where `(d1, d2)` are the slow/fast pixel indices, reordered for the
//! detector `orientation` (`_reorder_indexes_from_orientation`). The half-pixel
//! offset puts the position at the pixel *centre*.
//!
//! **dtype matters for bit-exactness.** pyFAI computes positions in whatever
//! dtype the index grid carries: `position_array` uses `numpy.fromfunction(...,
//! dtype=float64)` (→ f64 centres, feeding the geometry transform), while
//! `solidAngleArray` uses `dtype=float32` (→ f32 centres). numpy's weak scalar
//! promotion means the `+ 0.5`, `* pixel`, and later `- poni` all stay in the
//! grid's dtype. So this module exposes both [`Detector::centers_f64`] and
//! [`Detector::centers_f32`], each reproducing that path exactly.

use ndarray::Array2;

/// A flat, contiguous module detector (Pilatus / Eiger / generic).
///
/// Distances are in metres. `shape` is `(slow, fast)` = `(dim1/Y, dim2/X)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Detector {
    /// Detector name (matches pyFAI's class name where applicable).
    pub name: &'static str,
    /// Pixel size along the slow dimension (dim1, Y), in metres.
    pub pixel1: f64,
    /// Pixel size along the fast dimension (dim2, X), in metres.
    pub pixel2: f64,
    /// `(slow, fast)` pixel count.
    pub shape: (usize, usize),
    /// Detector orientation (pyFAI `Orientation`, 0–4). Pilatus/Eiger default 3.
    pub orientation: i32,
    /// `(slow, fast)` module size in pixels, or `None` for a gapless detector.
    pub module_size: Option<(usize, usize)>,
    /// `(slow, fast)` inter-module gap in pixels, or `None`.
    pub module_gap: Option<(usize, usize)>,
    /// Dead/gap-pixel sentinel value (`Detector.DUMMY`), or `None` if the
    /// detector defines none. Pilatus marks dead/gap pixels `-2`.
    pub dummy: Option<f64>,
    /// Tolerance around [`dummy`](Self::dummy) (`Detector.DELTA_DUMMY`), or
    /// `None` for an exact match. Pilatus uses `±1.5`.
    pub delta_dummy: Option<f64>,
}

impl Detector {
    /// Pilatus1M (`_dectris.py`): 172 µm pixels, 1043×981, module 195×487 gap
    /// 17×7, orientation 3.
    pub fn pilatus1m() -> Self {
        Detector {
            name: "Pilatus1M",
            pixel1: 172e-6,
            pixel2: 172e-6,
            shape: (1043, 981),
            orientation: 3,
            module_size: Some((195, 487)),
            module_gap: Some((17, 7)),
            dummy: Some(-2.0),
            delta_dummy: Some(1.5),
        }
    }

    /// Eiger4M (`_dectris.py`): 75 µm pixels, 2167×2070, module 514×1030 gap
    /// 37×10, orientation 3.
    pub fn eiger4m() -> Self {
        Detector {
            name: "Eiger4M",
            pixel1: 75e-6,
            pixel2: 75e-6,
            shape: (2167, 2070),
            orientation: 3,
            module_size: Some((514, 1030)),
            module_gap: Some((37, 10)),
            // The Eiger classes in `_dectris.py` do not override DUMMY, so they
            // inherit the base `Detector.DUMMY = None` (no sentinel).
            dummy: None,
            delta_dummy: None,
        }
    }

    /// Generic gapless detector with the given pixel sizes (m) and `(slow,
    /// fast)` shape, orientation 0.
    pub fn generic(pixel1: f64, pixel2: f64, shape: (usize, usize)) -> Self {
        Detector {
            name: "Detector",
            pixel1,
            pixel2,
            shape,
            orientation: 0,
            module_size: None,
            module_gap: None,
            dummy: None,
            delta_dummy: None,
        }
    }

    /// Number of pixels (`shape.0 * shape.1`).
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// The reordered float index `(d1, d2)` for pixel `(i, j)` under the
    /// detector orientation, port of `_reorder_indexes_from_orientation`
    /// (`center=True`). The values are exact integers, so the result is exact
    /// in both f32 and f64.
    ///
    /// Only orientations 0 and 3 (no reorder) are validated against golden
    /// data; 1/2/4 panic until a golden dataset exercises them, rather than
    /// shipping unverified arithmetic.
    #[inline]
    fn reordered_index(&self, i: usize, j: usize) -> (f64, f64) {
        match self.orientation {
            0 | 3 => (i as f64, j as f64),
            o => panic!(
                "Detector::reordered_index: orientation {o} not yet ported \
                 (no golden dataset exercises it); see _reorder_indexes_from_orientation"
            ),
        }
    }

    /// Raw pixel-centre positions `(p1, p2)` in metres as flat row-major `f64`
    /// vectors — the `position_array` path (`numpy.fromfunction(dtype=float64)`).
    /// PONI is **not** subtracted here (the geometry layer does that).
    pub fn centers_f64(&self) -> (Vec<f64>, Vec<f64>) {
        let (s0, s1) = self.shape;
        let mut p1 = Vec::with_capacity(self.size());
        let mut p2 = Vec::with_capacity(self.size());
        for i in 0..s0 {
            for j in 0..s1 {
                let (d1, d2) = self.reordered_index(i, j);
                p1.push(self.pixel1 * (d1 + 0.5));
                p2.push(self.pixel2 * (d2 + 0.5));
            }
        }
        (p1, p2)
    }

    /// Raw pixel-centre positions `(p1, p2)` as flat row-major `f32` vectors —
    /// the `solidAngleArray` path (`numpy.fromfunction(dtype=float32)`). Every
    /// operation is in `f32`, matching numpy's weak scalar promotion (the
    /// python-float `pixel`/`0.5` adopt the f32 array dtype).
    pub fn centers_f32(&self) -> (Vec<f32>, Vec<f32>) {
        let (s0, s1) = self.shape;
        let pixel1 = self.pixel1 as f32;
        let pixel2 = self.pixel2 as f32;
        let mut p1 = Vec::with_capacity(self.size());
        let mut p2 = Vec::with_capacity(self.size());
        for i in 0..s0 {
            for j in 0..s1 {
                let (d1, d2) = self.reordered_index(i, j);
                // numpy: pixel(weak f64) * (f32 + 0.5) -> done in f32.
                p1.push(pixel1 * (d1 as f32 + 0.5));
                p2.push(pixel2 * (d2 as f32 + 0.5));
            }
        }
        (p1, p2)
    }

    /// Pixel **corner** positions `(p1, p2)` in metres as flat row-major `f64`
    /// vectors over the `(shape.0 + 1, shape.1 + 1)` grid — the corner-grid path
    /// `calc_cartesian_positions(center=False)` that `corner_array` uses for a
    /// contiguous detector (`corner_array` builds the index grids with
    /// `numpy.arange(shape + 1.0)`, i.e. f64). No half-pixel offset: corner
    /// `(i, j)` is at `pixel * index`. PONI is not subtracted here.
    ///
    /// The flat detectors in scope are contiguous, so this is the corner
    /// primitive feeding the geometry transform. Mapping these corners into a
    /// radial/azimuthal `corner_array` (`_geometry.calc_rad_azim`, with chi
    /// discontinuity handling) is deferred to M6 (full pixel splitting).
    pub fn corner_positions_f64(&self) -> (Vec<f64>, Vec<f64>) {
        let (s0, s1) = self.shape;
        let (c0, c1) = (s0 + 1, s1 + 1);
        let mut p1 = Vec::with_capacity(c0 * c1);
        let mut p2 = Vec::with_capacity(c0 * c1);
        for i in 0..c0 {
            for j in 0..c1 {
                // orientation 0/3: corner index == pixel index (no reorder);
                // 1/2/4 reorder uses `shape` (not `shape-1`) for corners — not
                // yet ported (no golden exercises it).
                let (d1, d2) = match self.orientation {
                    0 | 3 => (i as f64, j as f64),
                    o => panic!("Detector::corner_positions_f64: orientation {o} not yet ported"),
                };
                p1.push(self.pixel1 * d1);
                p2.push(self.pixel2 * d2);
            }
        }
        (p1, p2)
    }

    /// The static module-gap mask (`_Dectris.calc_mask`): `1` over inter-module
    /// gap rows/columns, `0` elsewhere. `None` for a gapless detector.
    pub fn calc_mask(&self) -> Option<Array2<i8>> {
        let (ms0, ms1) = self.module_size?;
        let (mg0, mg1) = self.module_gap?;
        let (s0, s1) = self.shape;
        let mut mask = Array2::<i8>::zeros((s0, s1));
        // dim0 = Y: gap rows after each module.
        let mut i = ms0;
        while i < s0 {
            for r in i..(i + mg0).min(s0) {
                for c in 0..s1 {
                    mask[[r, c]] = 1;
                }
            }
            i += ms0 + mg0;
        }
        // dim1 = X: gap columns after each module.
        let mut k = ms1;
        while k < s1 {
            for c in k..(k + mg1).min(s1) {
                for r in 0..s0 {
                    mask[[r, c]] = 1;
                }
            }
            k += ms1 + mg1;
        }
        Some(mask)
    }

    /// The dummy value and tolerance the integrator feeds to preproc, as
    /// `data_t` (f32) — port of `_common.Detector.get_dummies`. pyFAI computes
    /// `actual_dummy = float32(image_dtype(int64(self.dummy)))`: the `int64`
    /// step truncates toward zero, the image-dtype step is lossless for an
    /// in-range integer dummy (the case for every detector in scope), so this
    /// reduces to `int64(dummy) as f32`. `delta_dummy` defaults to the float32
    /// machine epsilon when the detector leaves it unset (pyFAI's
    /// `numpy.finfo("float32").eps`). Returns `(None, None)` when the detector
    /// defines no dummy.
    pub fn get_dummies(&self) -> (Option<f32>, Option<f32>) {
        let Some(dummy) = self.dummy else {
            return (None, None);
        };
        let actual_dummy = (dummy as i64) as f32;
        let actual_delta = match self.delta_dummy {
            Some(dd) => dd as f32,
            None => f32::EPSILON,
        };
        (Some(actual_dummy), Some(actual_delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilatus1m_shape_and_pixel() {
        let d = Detector::pilatus1m();
        assert_eq!(d.shape, (1043, 981));
        assert_eq!(d.size(), 1_023_183);
        assert_eq!(d.pixel1, 172e-6);
        assert_eq!(d.orientation, 3);
    }

    #[test]
    fn center_is_half_pixel_offset() {
        // Pixel (0,0) centre is half a pixel from the corner: pixel*0.5.
        let d = Detector::generic(100e-6, 200e-6, (3, 4));
        let (p1, p2) = d.centers_f64();
        assert_eq!(p1[0], 100e-6 * 0.5);
        assert_eq!(p2[0], 200e-6 * 0.5);
        // Pixel (0,1): fast index 1 -> pixel2*(1.5).
        assert_eq!(p2[1], 200e-6 * 1.5);
    }

    #[test]
    fn corner_positions_and_center_consistency() {
        let d = Detector::generic(100e-6, 200e-6, (3, 4));
        let (cp1, cp2) = d.corner_positions_f64();
        // corner grid is (shape+1) x (shape+1).
        assert_eq!(cp1.len(), 4 * 5);
        // corner (0,0) at the origin; corner (1,1) at one pixel.
        assert_eq!(cp1[0], 0.0);
        assert_eq!(cp2[0], 0.0);
        let cidx = |i: usize, j: usize| i * 5 + j;
        assert_eq!(cp1[cidx(1, 0)], 100e-6);
        assert_eq!(cp2[cidx(0, 1)], 200e-6);
        // Pixel (0,0) centre = mean of its 4 corners (A,B,C,D) = pixel*0.5.
        let (p1, _) = d.centers_f64();
        let mean1 = (cp1[cidx(0, 0)] + cp1[cidx(1, 0)] + cp1[cidx(1, 1)] + cp1[cidx(0, 1)]) / 4.0;
        assert_eq!(mean1, p1[0]);
    }

    #[test]
    fn pilatus_dummies_are_minus2_pm_1_5() {
        // Pilatus marks dead/gap pixels at -2 with a ±1.5 tolerance; get_dummies
        // returns them as f32 (the data_t the preproc engine consumes).
        let (dummy, delta) = Detector::pilatus1m().get_dummies();
        assert_eq!(dummy, Some(-2.0f32));
        assert_eq!(delta, Some(1.5f32));
        // A gapless generic detector defines no sentinel.
        assert_eq!(
            Detector::generic(1e-4, 1e-4, (2, 2)).get_dummies(),
            (None, None)
        );
    }

    #[test]
    fn module_gap_mask_marks_gap_rows() {
        let d = Detector::pilatus1m();
        let mask = d.calc_mask().expect("pilatus has gaps");
        // First gap row block starts at module_size.0 = 195.
        assert_eq!(mask[[194, 0]], 0); // last row of module 1
        assert_eq!(mask[[195, 0]], 1); // first gap row
        assert_eq!(mask[[211, 0]], 1); // 195+17-1 = 211, last gap row
        assert_eq!(mask[[212, 0]], 0); // first row of module 2
    }
}
