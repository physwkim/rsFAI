//! Reflection (systematic-absence) selection rules, ported from
//! `pyFAI/crystallography/space_groups.py` (`ReflectionCondition`).
//!
//! Only the centering-type rules wired by `Cell.type` are needed here:
//! `default` (P) plus the A/B/C/F/I/R centerings. Each takes a Miller triplet
//! and returns `true` when the reflection is allowed by symmetry. pyFAI's full
//! per-space-group table (5000+ lines, mostly AI-generated and flagged as
//! unvalidated upstream) is out of scope: the shipped calibrants are built from
//! the lattice-type rules plus, for diamond-FCC, one extra extinction rule
//! (see `cell::Cell::diamond`).

/// Default rule (`ReflectionCondition.default`, also `type_P`):
/// `h == k == l == 0` is forbidden.
pub fn default(h: i64, k: i64, l: i64) -> bool {
    !(h == 0 && k == 0 && l == 0)
}

/// End-centered A: `k + l` even.
pub fn type_a(_h: i64, k: i64, l: i64) -> bool {
    (k + l).rem_euclid(2) == 0
}

/// End-centered B: `h + l` even.
pub fn type_b(h: i64, _k: i64, l: i64) -> bool {
    (h + l).rem_euclid(2) == 0
}

/// End-centered C: `h + k` even.
pub fn type_c(h: i64, k: i64, _l: i64) -> bool {
    (h + k).rem_euclid(2) == 0
}

/// Face-centered F: `h, k, l` all even or all odd.
pub fn type_f(h: i64, k: i64, l: i64) -> bool {
    let s = h.rem_euclid(2) + k.rem_euclid(2) + l.rem_euclid(2);
    s == 0 || s == 3
}

/// Body-centered I: `h + k + l` even.
pub fn type_i(h: i64, k: i64, l: i64) -> bool {
    (h + k + l).rem_euclid(2) == 0
}

/// Rhombohedral R: `-h + k + l` a multiple of 3.
pub fn type_r(h: i64, k: i64, l: i64) -> bool {
    (-h + k + l).rem_euclid(3) == 0
}

/// The centering types recognized by `Cell.type` (`pyFAI`'s `Cell.types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Centering {
    P,
    I,
    F,
    A,
    B,
    C,
    R,
}

impl Centering {
    /// Parse a single-letter centering code; unknown codes fall back to `P`
    /// (mirrors `Cell.type` setter: `lattice_type if in types else "P"`).
    pub fn from_letter(s: &str) -> Centering {
        match s {
            "I" => Centering::I,
            "F" => Centering::F,
            "A" => Centering::A,
            "B" => Centering::B,
            "C" => Centering::C,
            "R" => Centering::R,
            _ => Centering::P,
        }
    }

    /// The single-character code, for the `Cell` `space_group` string.
    pub fn letter(self) -> &'static str {
        match self {
            Centering::P => "P",
            Centering::I => "I",
            Centering::F => "F",
            Centering::A => "A",
            Centering::B => "B",
            Centering::C => "C",
            Centering::R => "R",
        }
    }

    /// The centering-specific rule (the rule beyond `default`), or `None` for
    /// primitive (`P`), which uses only `default`.
    pub fn rule(self) -> Option<fn(i64, i64, i64) -> bool> {
        match self {
            Centering::P => None,
            Centering::I => Some(type_i),
            Centering::F => Some(type_f),
            Centering::A => Some(type_a),
            Centering::B => Some(type_b),
            Centering::C => Some(type_c),
            Centering::R => Some(type_r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_forbids_origin() {
        assert!(!default(0, 0, 0));
        assert!(default(1, 0, 0));
        assert!(default(0, 0, -1));
    }

    #[test]
    fn fcc_rule() {
        assert!(type_f(1, 1, 1)); // all odd
        assert!(type_f(2, 0, 0)); // all even
        assert!(!type_f(1, 1, 0)); // mixed
        assert!(type_f(-1, -1, -1)); // all odd, negative
    }

    #[test]
    fn bcc_rule() {
        assert!(type_i(1, 1, 0));
        assert!(!type_i(1, 0, 0));
        assert!(type_i(-1, -1, 0));
    }
}
