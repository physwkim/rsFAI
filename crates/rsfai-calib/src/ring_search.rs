//! Dioptas's automatic refinement (Dioptas 0.13.0,
//! `CalibrationModel.search_peaks_on_ring` and
//! `CalibrationController.automatic_refinement`): from a start geometry,
//! pick the peaks in a narrow 2θ band around each calibrant ring in turn
//! and refine the geometry on every peak picked so far, so each ring is
//! searched where the geometry refined on the rings before it puts it.
//!
//! Parity: the band ([`ring_band`]) is Dioptas's to the threshold, whose
//! mean and spread are summed here in order, not numpy's pairwise order
//! (a last-bit difference in a cut on whole counts); the picking is
//! [`Massif`]'s, its visit order seeded rather than numpy's; and each
//! refinement is [`GeometryRefinement::refine_with_fixed`]'s Nelder-Mead,
//! not scipy's SLSQP. The picked peaks and the converged geometry are
//! therefore checked at a tolerance, not bit for bit.

use rsfai_calibrant::Calibrant;
use rsfai_detectors::Detector;
use rsfai_geometry::calc_pos_zyx;
use rsfai_peaks::Massif;

use crate::refinement::{point_cartesian, GeometryParams, GeometryRefinement, Param};

/// How a ring's band is cut out of the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingSearch {
    /// Half-width of the band around the ring, degrees (Dioptas's Δ2θ,
    /// 0.1).
    pub delta_tth_deg: f64,
    /// Only band pixels brighter than `factor · mean + std` of the band
    /// are searched (Dioptas's intensity mean factor, 1).
    pub min_mean_factor: f64,
    /// Pixels brighter than this are left out of the band and its
    /// statistics (Dioptas's intensity limit, 55000).
    pub upper_limit: f64,
}

impl Default for RingSearch {
    fn default() -> RingSearch {
        RingSearch {
            delta_tth_deg: 0.1,
            min_mean_factor: 1.0,
            upper_limit: 55000.0,
        }
    }
}

/// The pixels of a ring's band worth searching, and the cuts of the search
/// (`search_peaks_on_ring`'s `mask2`, `Imin` and `keep`).
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    pub area: Vec<bool>,
    /// A peak must be brighter than this: the band's mean less its spread.
    pub imin: f64,
    /// At most this many peaks: the square root of the area's size.
    pub keep: usize,
}

/// 2θ (rad) at every pixel centre under `param`, row-major
/// (`Geometry.twoThetaArray`): [`GeometryRefinement::tth`]'s chain over the
/// whole detector, a block of rows at a time.
pub fn tth_array(detector: &Detector, param: &GeometryParams) -> Vec<f64> {
    const ROWS: usize = 256;
    let (h, w) = detector.shape;
    let mut tth = Vec::with_capacity(h * w);
    for r0 in (0..h).step_by(ROWS) {
        let rows = r0..(r0 + ROWS).min(h);
        let (p1, p2): (Vec<f64>, Vec<f64>) = rows
            .flat_map(|i| (0..w).map(move |j| (i, j)))
            .map(|(i, j)| point_cartesian(detector, i as f64, j as f64))
            .unzip();
        let pos = calc_pos_zyx(
            param.dist,
            param.poni1,
            param.poni2,
            param.rot1,
            param.rot2,
            param.rot3,
            &p1,
            &p2,
            None,
            detector.orientation,
        );
        tth.extend(
            pos.y
                .iter()
                .zip(&pos.x)
                .zip(&pos.z)
                .map(|((&t1, &t2), &t3)| (t1 * t1 + t2 * t2).sqrt().atan2(t3)),
        );
    }
    tth
}

/// The band within `search.delta_tth_deg` of `tth_ring` on `image`, less the
/// pixels `mask` marks (true = invalid), cut to the pixels bright enough to
/// hold a peak (`search_peaks_on_ring` up to its `peaks_from_area` call).
/// An empty band has an empty area.
pub fn ring_band(
    image: &[f32],
    tth: &[f64],
    tth_ring: f64,
    mask: Option<&[bool]>,
    search: &RingSearch,
) -> Band {
    assert_eq!(tth.len(), image.len(), "tth shape");
    if let Some(m) = mask {
        assert_eq!(m.len(), image.len(), "mask shape");
    }
    // Dioptas's conversion, not `to_radians`: for some widths (0.148) the
    // two differ in the last bit.
    let delta = search.delta_tth_deg / 180.0 * std::f64::consts::PI;
    let band: Vec<bool> = tth
        .iter()
        .enumerate()
        .map(|(i, &t)| (t - tth_ring).abs() <= delta && !mask.is_some_and(|m| m[i]))
        .collect();
    // nanmean / nanstd over the band, the pixels above the limit and NaN
    // left out.
    let counted = || {
        band.iter()
            .zip(image)
            .filter(|&(&b, _)| b)
            .map(|(_, &v)| f64::from(v))
            .filter(|&v| v <= search.upper_limit)
    };
    let n = counted().count();
    if n == 0 {
        return Band {
            area: vec![false; image.len()],
            imin: f64::NAN,
            keep: 0,
        };
    }
    let mean = counted().sum::<f64>() / n as f64;
    let std = (counted().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    let threshold = search.min_mean_factor * mean + std;
    let area: Vec<bool> = band
        .iter()
        .zip(image)
        .map(|(&b, &v)| b && f64::from(v) > threshold && f64::from(v) <= search.upper_limit)
        .collect();
    let size = area.iter().filter(|&&a| a).count();
    Band {
        area,
        imin: mean - std,
        keep: (size as f64).sqrt().ceil() as usize,
    }
}

/// How [`auto_refine`] runs.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoRefine {
    pub search: RingSearch,
    /// Rings searched, from the innermost (Dioptas's number of rings).
    pub rings: usize,
    /// Parameters held at the start value; the wavelength always is.
    pub fixed: Vec<Param>,
    /// Fixes the order the peaks are picked in.
    pub seed: u64,
}

/// A refined geometry and the peaks it was refined on.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    pub param: GeometryParams,
    /// chi2 of `param` over `points`, rad².
    pub chi2: f64,
    /// `(y, x, ring)`: pixel coordinates and the calibrant ring index.
    pub points: Vec<(f64, f64, usize)>,
}

/// Dioptas's automatic refinement from `start`: the first `how.rings` rings
/// of `calibrant` (fewer when the calibrant has fewer at the start
/// wavelength) searched in turn, and the geometry refined on every peak
/// found so far after each ring from the second on.
pub fn auto_refine(
    image: &[f32],
    mask: Option<&[bool]>,
    calibrant: &Calibrant,
    detector: &Detector,
    start: GeometryParams,
    how: &AutoRefine,
) -> Calibration {
    let (h, w) = detector.shape;
    assert_eq!(image.len(), h * w, "image shape");
    let mut calibrant = calibrant.clone();
    calibrant.set_wavelength(start.wavelength);
    let massif = Massif::new(image, h, w, None);
    let mut param = start;
    let mut tth = tth_array(detector, &param);
    let mut points: Vec<(f64, f64, usize)> = Vec::new();
    for (ring, &tth_ring) in calibrant.get_2th().iter().take(how.rings).enumerate() {
        let band = ring_band(image, &tth, tth_ring, mask, &how.search);
        // Dioptas reads pyFAI's IndexError as a ring with no peaks.
        let found = massif
            .peaks_from_area(&band.area, band.imin, band.keep, 0.0, how.seed)
            .unwrap_or_default();
        points.extend(found.into_iter().map(|(y, x)| (y, x, ring)));
        if ring >= 1 && !points.is_empty() {
            let mut refinement =
                GeometryRefinement::new(points.clone(), calibrant.clone(), detector.clone(), param);
            refinement.refine_with_fixed(&how.fixed);
            param = refinement.param();
            tth = tth_array(detector, &param);
        }
    }
    let chi2 = if points.is_empty() {
        f64::NAN
    } else {
        GeometryRefinement::new(points.clone(), calibrant, detector.clone(), param).chi2_current()
    };
    Calibration {
        param,
        chi2,
        points,
    }
}
