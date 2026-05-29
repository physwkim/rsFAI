//! `ControlPoints` / `PointGroup`, ported from `pyFAI/control_points.py`.
//!
//! A `ControlPoints` instance is the container produced by manual or automatic
//! ring picking: an ordered set of `PointGroup`s, each holding a list of
//! `(y, x)` pixel coordinates that all lie on the same Debye-Scherrer ring of a
//! calibrant. The container carries the calibrant (and hence the ring 2theta
//! list) so it can hand the geometry-refinement layer the rows it needs.
//!
//! This is a pure data container: pushing groups, listing labels, and emitting
//! the `(y, x, ring)` / `(y, x, 2theta)` row lists that `GeometryRefinement`
//! consumes. All coordinates are `f64` (pyFAI stores them as Python floats and
//! refines in float64). There is no arithmetic here beyond the calibrant 2theta
//! lookup (`getList2theta`), so the container is bit-exact by construction; the
//! 2theta values themselves come from `rsfai-calibrant` (`Calibrant::get_2th`).
//!
//! Not ported (GUI / IO / interactive concerns out of the integration core):
//! matplotlib annotate/plot handles, the file `load`/`save` text format, the
//! `readRingNrFromKeyboard` prompt, and the deprecated property aliases. The
//! `PointGroup` label scheme (`get_label`: a, b, ... z, aa, ab, ...) is ported
//! because the group ordering in `get_labels`/`getList` depends on the per-group
//! `code` (`PointGroup.code`, `control_points.py:566-571`).

/// A group of control points on one Debye-Scherrer ring, `control_points.PointGroup`.
#[derive(Debug, Clone)]
pub struct PointGroup {
    /// Pixel coordinates `(y, x)` (slow, fast), `PointGroup.points`. pyFAI stores
    /// each point as `[y, x]`; the row emitters preserve that `(pt[0], pt[1])`
    /// order verbatim.
    pub points: Vec<(f64, f64)>,
    /// Ring index into the calibrant d-spacing list, or `None`, `PointGroup._ring`.
    pub ring: Option<usize>,
    /// String label (`a`, `b`, ... `z`, `aa`, ...), `PointGroup.label`.
    pub label: String,
    /// Numerical label used for sorting, `PointGroup.code`.
    pub code: usize,
}

/// Build the `(label, code)` for a 0-based group code, `PointGroup.get_label`
/// (`control_points.py:467-482`). Reproduced exactly: `code < 26` is one letter;
/// `code < 26*26` is two; otherwise three.
fn label_for_code(code: usize) -> String {
    if code < 26 {
        // chr(97 + code)
        ((97 + code) as u8 as char).to_string()
    } else if code < 26 * 26 {
        // chr(96 + code // 26) + chr(97 + code % 26)
        let a = (96 + code / 26) as u8 as char;
        let b = (97 + code % 26) as u8 as char;
        format!("{a}{b}")
    } else {
        // chr(96 + b // 26) + chr(97 + b % 26) + chr(97 + a), with a = code % 26,
        // b = code // 26.
        let a = code % 26;
        let b = code / 26;
        let c0 = (96 + b / 26) as u8 as char;
        let c1 = (97 + b % 26) as u8 as char;
        let c2 = (97 + a) as u8 as char;
        format!("{c0}{c1}{c2}")
    }
}

/// An ordered set of control-point groups bound to a calibrant 2theta list,
/// `control_points.ControlPoints`.
///
/// Groups are stored in insertion order (pyFAI uses an `OrderedDict` keyed by
/// label); `get_labels` returns them sorted by `code`, which for sequential
/// appends is the same as insertion order. The 2theta list is supplied directly
/// (the caller computes it from a `rsfai_calibrant::Calibrant`), keeping this
/// container free of any transcendental math.
#[derive(Debug, Clone, Default)]
pub struct ControlPoints {
    groups: Vec<PointGroup>,
    /// Next code to assign, mirroring `PointGroup.last_label` (per-instance here,
    /// not a class global — each container owns its own counter).
    next_code: usize,
}

impl ControlPoints {
    /// An empty container.
    pub fn new() -> ControlPoints {
        ControlPoints::default()
    }

    /// Append a group of `(y, x)` points on the given ring, `ControlPoints.append`
    /// (`control_points.py:115-127`). Returns the assigned label.
    pub fn append(&mut self, points: Vec<(f64, f64)>, ring: Option<usize>) -> String {
        let code = self.next_code;
        self.next_code += 1;
        let label = label_for_code(code);
        let group = PointGroup {
            points,
            ring,
            label: label.clone(),
            code,
        };
        self.groups.push(group);
        label
    }

    /// Number of groups, `ControlPoints.__len__`.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether the container holds no groups.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// The groups in insertion order.
    pub fn groups(&self) -> &[PointGroup] {
        &self.groups
    }

    /// Labels sorted by `code`, `ControlPoints.get_labels` (`control_points.py:440-447`).
    pub fn get_labels(&self) -> Vec<String> {
        let mut idx: Vec<&PointGroup> = self.groups.iter().collect();
        idx.sort_by_key(|g| g.code);
        idx.into_iter().map(|g| g.label.clone()).collect()
    }

    /// The control points as `(y, x, ring)` rows, `ControlPoints.getListRing`
    /// (a.k.a. `getList`, `control_points.py:334-343`). Every point of every group
    /// is emitted with its group's ring index; groups with no ring are skipped
    /// (pyFAI's `gpt.ring` would be `None`, which the geometry layer cannot use).
    pub fn list_ring(&self) -> Vec<(f64, f64, usize)> {
        let mut out = Vec::new();
        for g in &self.groups {
            if let Some(ring) = g.ring {
                for &(y, x) in &g.points {
                    out.push((y, x, ring));
                }
            }
        }
        out
    }

    /// The control points as `(y, x, 2theta)` rows, `ControlPoints.getList2theta`
    /// (`control_points.py:322-332`). `tth` is the calibrant's visible-ring 2theta
    /// list (`Calibrant::get_2th`, radians); a group whose ring index is `>=
    /// tth.len()` is skipped, exactly as pyFAI's `if gpt.ring < len(tth)` guard.
    /// The emitted 2theta is `tth[ring]` verbatim (no arithmetic here), so the
    /// rows are bit-exact given a bit-exact `tth`.
    pub fn list_2theta(&self, tth: &[f64]) -> Vec<(f64, f64, f64)> {
        let mut out = Vec::new();
        for g in &self.groups {
            if let Some(ring) = g.ring {
                if ring < tth.len() {
                    let tthi = tth[ring];
                    for &(y, x) in &g.points {
                        out.push((y, x, tthi));
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_scheme_matches_pyfai() {
        // chr(97+0)='a', chr(97+25)='z'; then 'aa','ab',...
        assert_eq!(label_for_code(0), "a");
        assert_eq!(label_for_code(25), "z");
        // code 26: 96 + 26//26 = 97 -> 'a'; 97 + 26%26 = 97 -> 'a' => "aa".
        assert_eq!(label_for_code(26), "aa");
        // code 27: 'a','b' => "ab".
        assert_eq!(label_for_code(27), "ab");
    }

    #[test]
    fn append_assigns_sequential_labels_and_codes() {
        let mut cp = ControlPoints::new();
        let l0 = cp.append(vec![(1.0, 2.0)], Some(0));
        let l1 = cp.append(vec![(3.0, 4.0), (5.0, 6.0)], Some(1));
        assert_eq!(l0, "a");
        assert_eq!(l1, "b");
        assert_eq!(cp.len(), 2);
        assert_eq!(cp.get_labels(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn list_ring_preserves_y_x_ring_order() {
        let mut cp = ControlPoints::new();
        cp.append(vec![(10.0, 20.0), (30.0, 40.0)], Some(2));
        cp.append(vec![(50.0, 60.0)], None); // no ring -> skipped
        let rows = cp.list_ring();
        assert_eq!(rows, vec![(10.0, 20.0, 2), (30.0, 40.0, 2)]);
    }

    #[test]
    fn list_2theta_skips_out_of_range_rings() {
        let mut cp = ControlPoints::new();
        cp.append(vec![(1.0, 1.0)], Some(0));
        cp.append(vec![(2.0, 2.0)], Some(5)); // ring 5 not in a 2-entry tth -> skipped
        let tth = [0.5_f64, 0.7];
        let rows = cp.list_2theta(&tth);
        assert_eq!(rows, vec![(1.0, 1.0, 0.5)]);
    }
}
