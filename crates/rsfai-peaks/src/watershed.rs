//! Inverse watershed peak finder, a port of `pyFAI/ext/watershed.pyx`'s
//! `InverseWatershed`.
//!
//! The pipeline (`watershed.pyx:260`):
//!   1. `init_labels` — every pixel hill-climbs (via the bilinear discrete
//!      `c_local_maxi`) to a local-maximum pixel; that maximum's flat index is
//!      the pixel's label. Each maximum seeds a [`Region`].
//!   2. `init_borders` — per-pixel 8-bit code of which neighbours have a
//!      different label.
//!   3. `init_regions` — accumulate region sizes and, for border pixels, append
//!      the first differing neighbour's label (in the fixed bit-priority order
//!      `watershed.pyx:333`).
//!   4. `init_pass` — per region, find the highest border ("pass") value and the
//!      neighbour it passes to; drop regions whose neighbour/border counts
//!      disagree.
//!
//! `pyFAI.init()` stops here (the `merge_*` steps are commented out), so the
//! port stops here too. [`InverseWatershed::peaks_from_area`] then extracts the
//! peak coordinates (`watershed.pyx:485`).
//!
//! All of this is deterministic (integer labels / index arithmetic, `f32`
//! values), so the extracted peak coordinates are bit-exact.

use std::collections::BTreeMap;

use crate::bilinear::Bilinear;

/// A watershed region: a peak pixel, its catchment size, its border pixels and
/// the neighbouring region each border pixel touches, and the pass values.
/// Mirrors the Cython `Region` (`watershed.pyx:57`).
#[derive(Debug, Clone)]
pub struct Region {
    pub index: i32,
    pub size: i32,
    pub pass_to: i32,
    pub mini: f32,
    pub maxi: f32,
    pub highest_pass: f32,
    pub neighbors: Vec<i32>,
    pub border: Vec<i32>,
    pub peaks: Vec<i32>,
}

impl Region {
    fn new(idx: i32) -> Self {
        Region {
            index: idx,
            size: 0,
            pass_to: -1,
            mini: -1.0,
            maxi: -1.0,
            highest_pass: -(i64::MAX as f32), // -sys.maxsize, watershed.pyx:72
            neighbors: Vec::new(),
            border: vec![],
            peaks: vec![idx],
        }
    }

    /// `init_values` (`watershed.pyx:84`): set maxi/mini/highest_pass/pass_to
    /// from the border values. Returns `true` if the region is degenerate and
    /// must be dropped (neighbour/border count mismatch or no border).
    fn init_values(&mut self, flat: &[f32]) -> bool {
        self.maxi = flat[self.index as usize];
        let border_size = self.border.len();
        let neighbors_size = self.neighbors.len();
        if neighbors_size != border_size {
            return true;
        }
        if neighbors_size == 0 {
            return true;
        }
        let mut imax = 0usize;
        let mut i = self.border[imax] as usize;
        let mut mini = flat[i];
        let mut maxi = flat[i];
        for k in 1..border_size {
            i = self.border[k] as usize;
            let val = flat[i];
            if val < mini {
                mini = val;
            } else if val > maxi {
                maxi = val;
                imax = k;
            }
        }
        if self.mini == -1.0 {
            self.mini = mini;
        }
        self.highest_pass = maxi;
        self.pass_to = self.neighbors[imax];
        false
    }
}

#[inline]
fn get_bit(byteval: u8, idx: u32) -> bool {
    (byteval & (1 << idx)) != 0
}

/// The inverse-watershed segmenter over a 2-D `f32` image.
pub struct InverseWatershed {
    data: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub labels: Vec<i32>,
    pub borders: Vec<u8>,
    /// Regions keyed by peak flat-index. After `init`, degenerate regions are
    /// removed. A `BTreeMap` gives the deterministic ascending key order that
    /// matches the Python `dict`'s insertion (peaks are inserted in raster
    /// order, which is ascending flat-index).
    pub regions: BTreeMap<i32, Region>,
}

impl InverseWatershed {
    /// Wrap a row-major `f32` image of shape `(height, width)`.
    pub fn new(data: Vec<f32>, height: usize, width: usize) -> Self {
        assert_eq!(
            data.len(),
            height * width,
            "data length must be height*width"
        );
        InverseWatershed {
            data,
            height,
            width,
            labels: vec![0; height * width],
            borders: vec![0; height * width],
            regions: BTreeMap::new(),
        }
    }

    /// Run the deterministic `init` pipeline (labels, borders, regions, pass).
    pub fn init(&mut self) {
        self.init_labels();
        self.init_borders();
        self.init_regions();
        self.init_pass();
    }

    fn init_labels(&mut self) {
        let bilinear = Bilinear::new(&self.data, self.height, self.width);
        for i in 0..self.height {
            for j in 0..self.width {
                let idx = j + i * self.width;
                let res = bilinear.local_maxi_index(idx) as i32;
                self.labels[idx] += res;
                if idx as i32 == res {
                    self.regions.insert(res, Region::new(res));
                }
            }
        }
    }

    fn init_borders(&mut self) {
        let h = self.height as i64;
        let w = self.width as i64;
        for i in 0..self.height as i64 {
            for j in 0..self.width as i64 {
                let mut neighb: u8 = 0;
                let res = self.labels[(j + i * w) as usize];
                let lab = |r: i64, c: i64| self.labels[(c + r * w) as usize];
                if i > 0 && j > 0 && lab(i - 1, j - 1) != res {
                    neighb |= 1;
                }
                if i > 0 && lab(i - 1, j) != res {
                    neighb |= 1 << 1;
                }
                if i > 0 && j < w - 1 && lab(i - 1, j + 1) != res {
                    neighb |= 1 << 2;
                }
                if j < w - 1 && lab(i, j + 1) != res {
                    neighb |= 1 << 3;
                }
                if i < h - 1 && j < w - 1 && lab(i + 1, j + 1) != res {
                    neighb |= 1 << 4;
                }
                if i < h - 1 && lab(i + 1, j) != res {
                    neighb |= 1 << 5;
                }
                if i < h - 1 && j > 0 && lab(i + 1, j - 1) != res {
                    neighb |= 1 << 6;
                }
                if j > 0 && lab(i, j - 1) != res {
                    neighb |= 1 << 7;
                }
                self.borders[(j + i * w) as usize] = neighb;
            }
        }
    }

    fn init_regions(&mut self) {
        let w = self.width as i64;
        for i in 0..self.height as i64 {
            for j in 0..self.width as i64 {
                let idx = (j + i * w) as i32;
                let neighb = self.borders[(j + i * w) as usize];
                let res = self.labels[(j + i * w) as usize];
                let lab = |r: i64, c: i64| self.labels[(c + r * w) as usize];
                let region = self
                    .regions
                    .get_mut(&res)
                    .expect("every label is a region peak");
                region.size += 1;
                if neighb == 0 {
                    continue;
                }
                region.border.push(idx);
                // Fixed bit-priority order (watershed.pyx:333): orthogonal
                // neighbours (bits 1,3,5,7) before diagonals (0,2,4,6).
                let nb = if get_bit(neighb, 1) {
                    lab(i - 1, j)
                } else if get_bit(neighb, 3) {
                    lab(i, j + 1)
                } else if get_bit(neighb, 5) {
                    lab(i + 1, j)
                } else if get_bit(neighb, 7) {
                    lab(i, j - 1)
                } else if get_bit(neighb, 0) {
                    lab(i - 1, j - 1)
                } else if get_bit(neighb, 2) {
                    lab(i - 1, j + 1)
                } else if get_bit(neighb, 4) {
                    lab(i + 1, j + 1)
                } else {
                    // bit 6
                    lab(i + 1, j - 1)
                };
                region.neighbors.push(nb);
            }
        }
    }

    fn init_pass(&mut self) {
        let flat = &self.data;
        let keys: Vec<i32> = self.regions.keys().copied().collect();
        for key in keys {
            let drop = {
                let region = self.regions.get_mut(&key).unwrap();
                region.init_values(flat)
            };
            if drop {
                self.regions.remove(&key);
            }
        }
    }

    /// Extract peak coordinates within `mask` (`peaks_from_area`,
    /// `watershed.pyx:485`).
    ///
    ///   * `mask`: row-major non-zero-is-valid flags (same shape as the image).
    ///   * `imin`: keep only peaks whose raw intensity is `>= imin`.
    ///   * `keep`: cap the number of returned peaks.
    ///   * `refine`: sub-pixel-refine each kept peak via the bilinear Taylor fit.
    ///   * `dmin`: minimum distance (in pixels) between kept peaks.
    ///
    /// Returns peaks as `(y, x)` `f32` pairs (refined) or integer-valued `f32`
    /// pairs (`refine == false`).
    ///
    /// **Ordering caveat.** The peak *coordinates* are bit-exact vs pyFAI, but
    /// the returned *list order* is not a portable contract: pyFAI gathers peaks
    /// region-by-region in the iteration order of a CPython `set` of (large)
    /// region flat-indices, whose hash-table slot order this `BTreeSet`
    /// (ascending) does not reproduce. When every kept peak has a distinct
    /// intensity the final stable intensity sort makes the order total and
    /// identical to pyFAI; only when intensities tie can the order differ while
    /// the coordinate set stays bit-exact. Callers that need the exact pyFAI
    /// order for tied intensities must treat the result as a set.
    pub fn peaks_from_area(
        &self,
        mask: &[bool],
        imin: Option<f32>,
        keep: Option<usize>,
        refine: bool,
        dmin: f32,
    ) -> Vec<(f32, f32)> {
        assert_eq!(mask.len(), self.height * self.width, "mask shape");
        let width = self.width;
        // Regions touched by any valid pixel. pyFAI builds a CPython `set`
        // (`keep_regions`) from `numpy.where(mask)` and later iterates it in
        // hash-table slot order; we collect the same membership into a
        // `BTreeSet` (ascending region index). This changes only the gather
        // order for tied-intensity peaks — see the ordering caveat on
        // `peaks_from_area` — not the set of coordinates produced.
        let mut keep_regions: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
        for (i, &m) in mask.iter().enumerate() {
            if m {
                let label = self.labels[i];
                if let Some(region) = self.regions.get(&label) {
                    keep_regions.insert(region.index);
                }
            }
        }

        let mut output_points: Vec<(f32, f32)> = Vec::new();
        let mut intensities: Vec<f32> = Vec::new();
        for ri in &keep_regions {
            let region = &self.regions[ri];
            for &j in &region.peaks {
                if mask[j as usize] {
                    intensities.push(self.data[j as usize]);
                    let x = j as usize % width;
                    let y = j as usize / width;
                    output_points.push((y as f32, x as f32));
                }
            }
        }

        if refine {
            let bilinear = Bilinear::new(&self.data, self.height, self.width);
            for p in output_points.iter_mut() {
                *p = bilinear.local_maxi(*p);
            }
        }

        if imin.is_some() || keep.is_some() {
            // argsort by intensity descending, stable on ties (matches Python's
            // `sorted(range(n), key=intensities.__getitem__, reverse=True)`,
            // which is stable, so equal intensities keep ascending index order).
            let mut argsort: Vec<usize> = (0..intensities.len()).collect();
            argsort.sort_by(|&a, &b| {
                intensities[b]
                    .partial_cmp(&intensities[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            if let Some(im) = imin {
                argsort.retain(|&i| intensities[i] >= im);
            }
            let mut ordered: Vec<(f32, f32)> = argsort.iter().map(|&i| output_points[i]).collect();

            let dmin2 = if dmin != 0.0 { dmin * dmin } else { 0.0 };
            if let Some(k) = keep {
                if ordered.len() > k {
                    let tmp_lst = ordered.clone();
                    let mut rej_lst: Vec<(f32, f32)> = Vec::new();
                    ordered = Vec::new();
                    for pt in tmp_lst {
                        let mut too_close = false;
                        for pt2 in &ordered {
                            let d2 = (pt.0 - pt2.0).powi(2) + (pt.1 - pt2.1).powi(2);
                            if d2 <= dmin2 {
                                too_close = true;
                                break;
                            }
                        }
                        if too_close {
                            rej_lst.push(pt);
                        } else {
                            ordered.push(pt);
                            if ordered.len() >= k {
                                return ordered;
                            }
                        }
                    }
                    ordered.extend(rej_lst);
                    ordered.truncate(k);
                }
            }
            return ordered;
        }
        output_points
    }
}
