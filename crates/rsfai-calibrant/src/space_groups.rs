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
//!
//! Two of pyFAI's per-space-group conditions are ported here because they back
//! shipped calibrants and are flagged `validated` upstream: `group166_r3bar_m`
//! (R-3̄m, hydrocerussite) and `group167_r3bar_c` (R-3̄c, the corundum-type
//! Cr2O3 / eskolaite). These are layered onto a primitive hexagonal `Cell` via
//! `Cell::add_selection_rule`, exactly as the pyFAI tutorials append
//! `ReflectionCondition.group16{6,7}_R3bar_{m,c}` to `Cell.selection_rules`.

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

/// Space group 166, R-3̄m (trigonal, hexagonal axes; e.g. hydrocerussite).
///
/// Verbatim port of `ReflectionCondition.group166_R3bar_m`. The first test is
/// the R-centring condition `-h + k + l = 3n`; the remaining branches are the
/// ITC special-position conditions in `(h, k, l)` (with `i = -(h + k)`).
/// Python's `%` on a positive modulus matches `i64::rem_euclid`.
pub fn group166_r3bar_m(h: i64, k: i64, l: i64) -> bool {
    if (-h + k + l).rem_euclid(3) != 0 {
        return false; // hkil
    }
    if l == 0 {
        return (-h + k).rem_euclid(3) == 0; // hki0
    }
    if h == k {
        return l.rem_euclid(3) == 0; // hh(-2h)l
    }
    if k == -h {
        return (h + l).rem_euclid(3) == 0; // h(-h)0l (i = 0)
    }
    if h == 0 && k == 0 {
        return l.rem_euclid(3) == 0; // 000l
    }
    if k == -h && l == 0 {
        return h.rem_euclid(3) == 0; // h(-h)00
    }
    true
}

/// Space group 167, R-3̄c (trigonal, hexagonal axes; the corundum-type Cr2O3 /
/// eskolaite calibrant).
///
/// Verbatim port of `ReflectionCondition.group167_R3bar_c`. Beyond the R-centring
/// `-h + k + l = 3n`, the c-glide tightens the special positions: `000l` requires
/// `l = 6n` (not `3n`), and `h(-h)0l` / `0kl` / `h0l` additionally require `l`
/// even. This is the extinction R-centring alone does not capture.
pub fn group167_r3bar_c(h: i64, k: i64, l: i64) -> bool {
    // (1) R-centring condition applies to all reflections.
    if (-h + k + l).rem_euclid(3) != 0 {
        return false;
    }
    // (5) 000l: l = 6n
    if h == 0 && k == 0 {
        return l.rem_euclid(6) == 0;
    }
    // (6) h(-h)00: h = 3n
    if k == -h && l == 0 {
        return h.rem_euclid(3) == 0;
    }
    // (2) hki0 (l = 0): -h + k = 3n
    if l == 0 {
        return (-h + k).rem_euclid(3) == 0;
    }
    // (3) hh(-2h)l: l = 3n
    if h == k {
        return l.rem_euclid(3) == 0;
    }
    // (4) h(-h)0l (i = 0): l = 2n and h + l = 3n
    if k == -h {
        return l.rem_euclid(2) == 0 && (h + l).rem_euclid(3) == 0;
    }
    // (7) 0kl (h = 0): l = 2n and k + l = 3n
    if h == 0 {
        return l.rem_euclid(2) == 0 && (k + l).rem_euclid(3) == 0;
    }
    // (8) h0l (k = 0): l = 2n and h - l = 3n
    if k == 0 {
        return l.rem_euclid(2) == 0 && (h - l).rem_euclid(3) == 0;
    }
    // (9) 0k0 (h = 0, l = 0): k = 3n
    if h == 0 && l == 0 {
        return k.rem_euclid(3) == 0;
    }
    true
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

    #[test]
    fn group166_matches_r_centring_on_000l() {
        // R-3m: 000l allowed when l = 3n, same as bare R-centring.
        assert!(group166_r3bar_m(0, 0, 3));
        assert!(group166_r3bar_m(0, 0, 6));
        assert!(!group166_r3bar_m(0, 0, 1));
        assert!(!group166_r3bar_m(0, 0, 2));
        // R-centring base condition still gates non-special reflections.
        assert!(!group166_r3bar_m(1, 0, 0)); // -h+k+l = -1
        assert!(group166_r3bar_m(1, 1, 0)); // -h+k+l = 0
    }

    #[test]
    fn group167_c_glide_tightens_000l_beyond_r_centring() {
        // The c-glide forbids 000l unless l = 6n — where bare R-centring (type_r)
        // and group 166 both allow l = 3n. This is the extinction the partial
        // R-centring path could not express.
        assert!(type_r(0, 0, 3)); // R-centring allows
        assert!(group166_r3bar_m(0, 0, 3)); // R-3m allows
        assert!(!group167_r3bar_c(0, 0, 3)); // R-3c forbids (l != 6n)
        assert!(group167_r3bar_c(0, 0, 6)); // l = 6n allowed
        assert!(!group167_r3bar_c(0, 0, 9)); // l = 9 not 6n
        assert!(group167_r3bar_c(0, 0, 12)); // l = 12 = 6n
    }

    #[test]
    fn group167_h_minus_h_0l_requires_even_l() {
        // h(-h)0l: with k = -h, the base condition -h+k+l = -2h+l = 3n implies
        // h+l = 3n (their difference is 3h), so the only discriminator left is
        // the c-glide's l-even rule. Pick reflections whose base condition passes.
        // (1,-1,2): base = 0, l even               -> allowed
        assert!(group167_r3bar_c(1, -1, 2));
        // (1,-1,5): base = 3 (passes), l odd        -> forbidden by c-glide,
        //           even though bare R-centring allows it.
        assert!(type_r(1, -1, 5)); // R-centring allows
        assert!(!group167_r3bar_c(1, -1, 5)); // c-glide forbids (l odd)
                                              // Base R-centring rejects -h+k+l != 3n outright.
        assert!(!group167_r3bar_c(1, 0, 0)); // -h+k+l = -1
    }
}
