//! Connected-component labelling and the exact Euclidean distance transform,
//! reproducing the deterministic `scipy.ndimage` primitives that
//! `pyFAI.massif.Massif` relies on (`massif.py:44`):
//!
//!   * [`label`] mirrors `scipy.ndimage.label(binarization, structure)` — the
//!     `get_labeled_massif` connected-component pass (`massif.py:365`). pyFAI
//!     passes `numpy.ones((3, 3), int8)` (8-connectivity); the structure is
//!     configurable here for the 4-connectivity case.
//!   * [`distance_transform_edt`] mirrors
//!     `scipy.ndimage.distance_transform_edt(mask, return_indices=True)` — the
//!     `cleaned_data` "second stage" mask in-fill (`massif.py:295`).
//!
//! Both are integer-deterministic: labels are `i32`, feature indices are `i32`,
//! and the squared-distance accumulation is integer, so the result is a pure
//! function of the input. The Rust output is gated bit-exact against the scipy
//! golden in `tests/golden_peaks.rs`.

/// A binary structuring element for connected-component labelling. The cells
/// mark which of the 8 neighbours (plus the centre, ignored) connect two
/// foreground pixels. `scipy.ndimage.generate_binary_structure(2, 2)` is the
/// 3x3 all-ones element (8-connectivity); `(2, 1)` is the plus shape
/// (4-connectivity).
#[derive(Debug, Clone, Copy)]
pub struct Structure {
    /// Row-major 3x3 connectivity mask; `cells[4]` (the centre) is unused.
    pub cells: [bool; 9],
}

impl Structure {
    /// 8-connectivity (`numpy.ones((3, 3))`), the pyFAI `get_labeled_massif`
    /// default (`massif.py:358`).
    pub fn full() -> Self {
        Structure { cells: [true; 9] }
    }

    /// 4-connectivity (the plus / cross structuring element).
    pub fn cross() -> Self {
        Structure {
            #[rustfmt::skip]
            cells: [
                false, true, false,
                true,  true, true,
                false, true, false,
            ],
        }
    }

    /// True iff the neighbour at relative offset `(dr, dc)` (each in `-1..=1`)
    /// connects to the centre.
    #[inline]
    fn connects(&self, dr: i32, dc: i32) -> bool {
        self.cells[((dr + 1) * 3 + (dc + 1)) as usize]
    }
}

/// Disjoint-set (union-find) with path compression and union-by-min so that the
/// representative of every set is the smallest provisional label in it — the
/// invariant `scipy.ndimage.label` uses to renumber components in raster order.
struct UnionFind {
    parent: Vec<i32>,
}

impl UnionFind {
    fn new() -> Self {
        // index 0 is the background sentinel; provisional labels start at 1.
        UnionFind { parent: vec![0] }
    }

    fn make(&mut self) -> i32 {
        let id = self.parent.len() as i32;
        self.parent.push(id);
        id
    }

    fn find(&mut self, mut x: i32) -> i32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    /// Union keeping the smaller root, returning that root.
    fn union(&mut self, a: i32, b: i32) -> i32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        self.parent[hi as usize] = lo;
        lo
    }
}

/// Connected-component labelling of a boolean `input` of shape `(rows, cols)`
/// (row-major), with the given connectivity `structure`. Returns the `i32`
/// label image (0 = background) and the number of components found, exactly as
/// `scipy.ndimage.label` does:
///
///   * components are numbered `1..=n` in the order their first (top-left,
///     raster-scan) pixel is reached;
///   * a foreground pixel takes the smallest label among already-labelled
///     connected neighbours, recording equivalences for a final renumber pass.
pub fn label(input: &[bool], rows: usize, cols: usize, structure: Structure) -> (Vec<i32>, i32) {
    assert_eq!(input.len(), rows * cols, "input length must be rows*cols");
    let mut prov = vec![0i32; rows * cols];
    let mut uf = UnionFind::new();

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if !input[idx] {
                continue;
            }
            // Smallest provisional root over the already-visited neighbours
            // (the four causal directions in raster order: NW, N, NE, W).
            let mut best: i32 = 0;
            // Visit in (dr, dc) order matching the causal half of the 3x3.
            for &(dr, dc) in &[(-1i32, -1i32), (-1, 0), (-1, 1), (0, -1)] {
                if !structure.connects(dr, dc) {
                    continue;
                }
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                    continue;
                }
                let nlab = prov[(nr as usize) * cols + (nc as usize)];
                if nlab == 0 {
                    continue;
                }
                let root = uf.find(nlab);
                if best == 0 || root < best {
                    best = root;
                }
            }
            if best == 0 {
                prov[idx] = uf.make();
            } else {
                // Merge every connected neighbour's set into `best`.
                for &(dr, dc) in &[(-1i32, -1i32), (-1, 0), (-1, 1), (0, -1)] {
                    if !structure.connects(dr, dc) {
                        continue;
                    }
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                        continue;
                    }
                    let nlab = prov[(nr as usize) * cols + (nc as usize)];
                    if nlab != 0 {
                        uf.union(best, nlab);
                    }
                }
                prov[idx] = best;
            }
        }
    }

    // Renumber roots consecutively in order of first appearance (raster scan),
    // matching scipy's final relabel.
    let mut remap = vec![0i32; uf.parent.len()];
    let mut next = 0i32;
    let mut out = vec![0i32; rows * cols];
    for idx in 0..rows * cols {
        let p = prov[idx];
        if p == 0 {
            continue;
        }
        let root = uf.find(p);
        if remap[root as usize] == 0 {
            next += 1;
            remap[root as usize] = next;
        }
        out[idx] = remap[root as usize];
    }
    (out, next)
}

/// Result of [`distance_transform_edt`]: for every pixel, the `(row, col)` index
/// of the nearest background (`false`) feature, plus the squared Euclidean
/// distance to it. Indices are `i32`; squared distances are exact `i64`.
pub struct EdtResult {
    /// Nearest-background row index per pixel (row-major), `scipy` `indices[0]`.
    pub idx_row: Vec<i32>,
    /// Nearest-background column index per pixel, `scipy` `indices[1]`.
    pub idx_col: Vec<i32>,
    /// Exact squared distance to the nearest background pixel (integer).
    pub dist2: Vec<i64>,
}

impl EdtResult {
    /// The Euclidean distance image (`sqrt` of [`dist2`]), as `scipy`'s
    /// `distance_transform_edt(..., return_distances=True)` returns it (f64).
    ///
    /// [`dist2`]: EdtResult::dist2
    pub fn distances(&self) -> Vec<f64> {
        self.dist2.iter().map(|&d| (d as f64).sqrt()).collect()
    }
}

/// Exact Euclidean distance transform with feature (nearest-background) indices,
/// reproducing `scipy.ndimage.distance_transform_edt(input, return_indices=True)`.
///
/// `input == true` marks foreground (the pixels whose distance to the nearest
/// `false` background is computed). Uses the Felzenszwalb–Huttenlocher separable
/// algorithm, one axis at a time, with the lower-envelope parabola sweep — the
/// same separable scheme `scipy`'s `euclidean_feature_transform` uses, so the
/// tie-breaking matches bit-for-bit (verified against scipy in the golden).
pub fn distance_transform_edt(input: &[bool], rows: usize, cols: usize) -> EdtResult {
    assert_eq!(input.len(), rows * cols, "input length must be rows*cols");
    const INF: i64 = i64::MAX / 4;

    // Pass 1 — along columns (axis 0): for each column, 1-D EDT giving, per
    // pixel, the nearest background ROW in that column.
    // feat_row[idx] = row of nearest background in the same column (or sentinel).
    let mut g = vec![INF; rows * cols]; // squared distance after axis-0 pass
    let mut feat_row = vec![-1i32; rows * cols];
    for c in 0..cols {
        // Lower-envelope 1-D transform over rows for this column. f(r) = 0 if
        // background else +inf; we want the nearest background row.
        // Simple two-pass (forward/backward) suffices for the 1-D axis since the
        // cost is f(r) + (r - r')^2 with f in {0, inf}; we track the source row.
        let mut last: i32 = -1;
        // forward
        for r in 0..rows {
            let idx = r * cols + c;
            if !input[idx] {
                last = r as i32;
            }
            if last >= 0 {
                let d = r as i64 - last as i64;
                g[idx] = d * d;
                feat_row[idx] = last;
            }
        }
        // backward
        let mut last: i32 = -1;
        for r in (0..rows).rev() {
            let idx = r * cols + c;
            if !input[idx] {
                last = r as i32;
            }
            if last >= 0 {
                let d = r as i64 - last as i64;
                let cand = d * d;
                if cand < g[idx] {
                    g[idx] = cand;
                    feat_row[idx] = last;
                }
            }
        }
    }

    // Pass 2 — along rows (axis 1): combine column results using the lower
    // envelope of parabolas, recovering the true 2-D nearest background.
    let mut dist2 = vec![INF; rows * cols];
    let mut out_row = vec![-1i32; rows * cols];
    let mut out_col = vec![-1i32; rows * cols];

    // Per-row scratch for the parabola sweep.
    let mut v = vec![0usize; cols]; // locations of parabola vertices
    let mut z = vec![0f64; cols + 1]; // boundaries between parabolas
    for r in 0..rows {
        let row_off = r * cols;
        // f(q) = g[row_off + q] (squared distance to nearest bg in column q).
        let f = |q: usize| g[row_off + q];
        let mut k: isize = 0;
        v[0] = 0;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;
        for q in 1..cols {
            if f(q) >= INF {
                // empty column at q: skip; it can never be a vertex with finite f.
                continue;
            }
            loop {
                if k < 0 {
                    k = 0;
                    v[0] = q;
                    z[0] = f64::NEG_INFINITY;
                    z[1] = f64::INFINITY;
                    break;
                }
                let p = v[k as usize];
                if f(p) >= INF {
                    // current vertex parabola is at infinity: replace it.
                    if k == 0 {
                        v[0] = q;
                        z[0] = f64::NEG_INFINITY;
                        z[1] = f64::INFINITY;
                        break;
                    }
                    k -= 1;
                    continue;
                }
                // intersection of parabolas from p and q.
                let s = ((f(q) - f(p)) as f64 + (q * q) as f64 - (p * p) as f64)
                    / (2.0 * (q as f64 - p as f64));
                if s <= z[k as usize] {
                    k -= 1;
                } else {
                    k += 1;
                    v[k as usize] = q;
                    z[k as usize] = s;
                    z[(k + 1) as usize] = f64::INFINITY;
                    break;
                }
            }
        }
        // Sweep q, picking the parabola whose interval contains q.
        let mut k2: isize = 0;
        for q in 0..cols {
            while (k2 as usize) < cols && z[(k2 + 1) as usize] < q as f64 {
                k2 += 1;
            }
            let p = v[k2 as usize];
            let out_idx = row_off + q;
            if f(p) >= INF {
                // no background found in this row band; leave sentinel
                dist2[out_idx] = INF;
                continue;
            }
            let dcol = q as i64 - p as i64;
            dist2[out_idx] = f(p) + dcol * dcol;
            out_row[out_idx] = feat_row[row_off + p];
            out_col[out_idx] = p as i32;
        }
    }

    EdtResult {
        idx_row: out_row,
        idx_col: out_col,
        dist2,
    }
}
