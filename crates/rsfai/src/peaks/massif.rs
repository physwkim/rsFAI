//! Peak picking inside an area, a port of `pyFAI.massif.Massif`'s
//! `nearest_peak` and `peaks_from_area` (`massif.py`) on the raw image
//! (`median_prefilter=False`, the way Dioptas builds it for its ring search).
//!
//! pyFAI visits the area's pixels in a `numpy.random.shuffle` order; that
//! order is not reproducible outside numpy, so the visit order is an input
//! of [`Massif::peaks_from_points`] (bit-exact against pyFAI given pyFAI's
//! order), and [`Massif::peaks_from_area`] shuffles with a seeded generator
//! of its own, so one image and seed always give the same peaks.

use crate::peaks::bilinear::{round_half_even, Bilinear};

/// An image wrapped for peak picking. Pixels that are masked, or not finite,
/// never yield a peak (`Massif.mask`, which pyFAI ORs with `~isfinite(data)`).
pub struct Massif<'a> {
    bilinear: Bilinear<'a>,
    data: &'a [f32],
    invalid: Option<Vec<bool>>,
}

impl<'a> Massif<'a> {
    /// Wrap a row-major `(height, width)` image; `mask` is true at invalid
    /// pixels.
    pub fn new(data: &'a [f32], height: usize, width: usize, mask: Option<&[bool]>) -> Self {
        if let Some(m) = mask {
            assert_eq!(m.len(), data.len(), "mask shape");
        }
        let nonfinite = data.iter().any(|v| !v.is_finite());
        let invalid = match (mask, nonfinite) {
            (None, false) => None,
            (m, _) => Some(
                data.iter()
                    .enumerate()
                    .map(|(i, v)| !v.is_finite() || m.is_some_and(|m| m[i]))
                    .collect(),
            ),
        };
        Massif {
            bilinear: Bilinear::new(data, height, width),
            data,
            invalid,
        }
    }

    /// The sub-pixel maximum the image climbs to from pixel `(y, x)`, or
    /// `None` when it lands on an invalid pixel (`Massif.nearest_peak`).
    pub fn nearest_peak(&self, y: usize, x: usize) -> Option<(f64, f64)> {
        let out = self.bilinear.local_maxi((y as f32, x as f32));
        let (h, w) = (self.bilinear.height as f64, self.bilinear.width as f64);
        if !(0.0 <= out.0 && out.0 < h && 0.0 <= out.1 && out.1 < w) {
            return None;
        }
        // `self.mask[int(res[0]), int(res[1])]`: int() truncates.
        let at = out.0 as usize * self.bilinear.width + out.1 as usize;
        match &self.invalid {
            Some(m) if m[at] => None,
            _ => Some(out),
        }
    }

    /// The peaks reached from `points`, visited in order
    /// (`Massif.peaks_from_area` after its shuffle): a peak is kept when it
    /// rounds to a pixel of `area` brighter than `imin` and lies farther than
    /// `dmin` from every peak kept before. The visit stops at `keep` peaks,
    /// or once more than `keep` points in a row added none. A peak within
    /// half a pixel of the far edge rounds off the image, where pyFAI's
    /// `mask[p0, p1]` raises `IndexError`; that ends the visit as
    /// [`OffImage`].
    pub fn peaks_from_points(
        &self,
        points: &[(usize, usize)],
        area: &[bool],
        imin: f64,
        keep: usize,
        dmin: f64,
    ) -> Result<Vec<(f64, f64)>, OffImage> {
        assert_eq!(area.len(), self.data.len(), "area shape");
        let (height, width) = (self.bilinear.height, self.bilinear.width);
        let dmin2 = dmin * dmin;
        let mut res: Vec<(f64, f64)> = Vec::new();
        let mut cnt = 0usize;
        for &(y, x) in points {
            if let Some(out) = self.nearest_peak(y, x) {
                let p0 = round_half_even(out.0) as usize;
                let p1 = round_half_even(out.1) as usize;
                if p0 >= height || p1 >= width {
                    return Err(OffImage(out));
                }
                let at = p0 * width + p1;
                if area[at] && f64::from(self.data[at]) > imin && far_from(out, &res, dmin2) {
                    res.push(out);
                    cnt = 0;
                }
            }
            if res.len() >= keep || cnt > keep {
                break;
            }
            cnt += 1;
        }
        Ok(res)
    }

    /// [`peaks_from_points`](Self::peaks_from_points) over every pixel of
    /// `area`, in an order shuffled by `seed`.
    pub fn peaks_from_area(
        &self,
        area: &[bool],
        imin: f64,
        keep: usize,
        dmin: f64,
        seed: u64,
    ) -> Result<Vec<(f64, f64)>, OffImage> {
        let width = self.bilinear.width;
        let mut points: Vec<(usize, usize)> = area
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .map(|(i, _)| (i / width, i % width))
            .collect();
        shuffle(&mut points, seed);
        self.peaks_from_points(&points, area, imin, keep, dmin)
    }
}

/// A peak that rounds to a pixel off the image (`IndexError` in pyFAI).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffImage(pub (f64, f64));

/// `is_far_from_group`: true when every kept peak is farther than
/// `sqrt(d2)`.
fn far_from(pt: (f64, f64), peaks: &[(f64, f64)], d2: f64) -> bool {
    peaks.iter().all(|&(q0, q1)| {
        let (d0, d1) = (pt.0 - q0, pt.1 - q1);
        d0 * d0 + d1 * d1 > d2
    })
}

/// Fisher–Yates over a SplitMix64 stream.
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two bumps: a visit from anywhere on either reaches its top.
    fn bumps() -> (Vec<f32>, usize, usize) {
        let (h, w) = (32, 32);
        let top =
            |y: f32, x: f32, cy: f32, cx: f32| (-((y - cy).powi(2) + (x - cx).powi(2)) / 8.0).exp();
        let data = (0..h * w)
            .map(|i| {
                let (y, x) = ((i / w) as f32, (i % w) as f32);
                100.0 * top(y, x, 8.0, 9.0) + 50.0 * top(y, x, 22.0, 20.0)
            })
            .collect();
        (data, h, w)
    }

    #[test]
    fn every_bump_is_found_once() {
        let (data, h, w) = bumps();
        let m = Massif::new(&data, h, w, None);
        let area = vec![true; h * w];
        let mut peaks = m.peaks_from_area(&area, 1.0, 10, 0.0, 7).unwrap();
        peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(peaks.len(), 2, "{peaks:?}");
        assert!((peaks[0].0 - 8.0).abs() < 0.1 && (peaks[0].1 - 9.0).abs() < 0.1);
        assert!((peaks[1].0 - 22.0).abs() < 0.1 && (peaks[1].1 - 20.0).abs() < 0.1);
    }

    #[test]
    fn a_peak_dimmer_than_imin_or_off_the_area_is_not_kept() {
        let (data, h, w) = bumps();
        let m = Massif::new(&data, h, w, None);
        let all = vec![true; h * w];
        assert_eq!(m.peaks_from_area(&all, 60.0, 10, 0.0, 7).unwrap().len(), 1);
        let top_half: Vec<bool> = (0..h * w).map(|i| i / w < 16).collect();
        let peaks = m.peaks_from_area(&top_half, 1.0, 10, 0.0, 7).unwrap();
        assert_eq!(peaks.len(), 1);
        assert!(peaks[0].0 < 16.0);
    }

    #[test]
    fn a_masked_top_yields_no_peak() {
        let (data, h, w) = bumps();
        let mut mask = vec![false; h * w];
        mask[8 * w + 9] = true;
        let m = Massif::new(&data, h, w, Some(&mask));
        assert_eq!(m.nearest_peak(10, 10), None);
        assert!(m.nearest_peak(20, 20).is_some());
    }

    #[test]
    fn the_seed_fixes_the_order() {
        let mut a: Vec<usize> = (0..100).collect();
        let mut b = a.clone();
        shuffle(&mut a, 3);
        shuffle(&mut b, 3);
        assert_eq!(a, b);
        assert_ne!(a, (0..100).collect::<Vec<_>>());
    }
}
