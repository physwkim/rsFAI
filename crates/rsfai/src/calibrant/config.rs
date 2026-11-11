//! `.D` d-spacing file parsing, ported from
//! `pyFAI/io/calibrant_config.py` (`CalibrantConfig.from_dspacing`) and the
//! `Miller` / `Reflection` containers from `pyFAI/containers.py`.
//!
//! A `.D` file is the historical pyFAI calibrant format: a header of `#`-prefixed
//! metadata (`Calibrant:`, `Cell:`, `Ref:`) followed by data lines of the shape
//!
//! ```text
//!  2.33797992 # (1 1 1)     8 100.0
//! ```
//!
//! i.e. `d_spacing # (h k l) multiplicity intensity`, with the Miller index,
//! multiplicity and intensity all optional. The only value the 2theta / peak
//! computation needs is the leading d-spacing (Angstrom); the rest is carried
//! for fidelity with pyFAI's `Reflection` list.

/// A family of lattice planes, `pyFAI.containers.Miller`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Miller {
    pub h: i64,
    pub k: i64,
    pub l: i64,
}

impl Miller {
    pub fn new(h: i64, k: i64, l: i64) -> Self {
        Self { h, k, l }
    }

    /// Port of `Miller.parse`: split on space/comma/semicolon, strip
    /// parentheses, parse exactly three integers.
    pub fn parse(text: &str) -> Option<Miller> {
        let mut ints: Vec<i64> = Vec::new();
        for word in text.split([' ', ',', ';']) {
            let stripped = word.trim_matches(|c| c == ' ' || c == '(' || c == ')');
            if stripped.is_empty() {
                continue;
            }
            // pyFAI logs a warning and skips non-integer tokens.
            if let Ok(v) = stripped.parse::<i64>() {
                ints.push(v);
            }
        }
        if ints.len() == 3 {
            Some(Miller::new(ints[0], ints[1], ints[2]))
        } else {
            None
        }
    }
}

/// One reflection (a family of Miller planes), `pyFAI.containers.Reflection`.
/// `dspacing` is in Angstrom; `intensity` / `hkl` / `multiplicity` are optional.
#[derive(Debug, Clone, PartialEq)]
pub struct Reflection {
    pub dspacing: f64,
    pub intensity: Option<f64>,
    pub hkl: Option<Miller>,
    pub multiplicity: Option<u32>,
}

impl Reflection {
    pub fn new(dspacing: f64) -> Self {
        Self {
            dspacing,
            intensity: None,
            hkl: None,
            multiplicity: None,
        }
    }
}

/// Parsed contents of a `.D` calibrant file, `pyFAI.io.calibrant_config.CalibrantConfig`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CalibrantConfig {
    pub name: String,
    pub description: String,
    pub cell: String,
    pub space_group: String,
    pub reference: String,
    pub reflections: Vec<Reflection>,
}

/// The text after the first `:`, trimmed — pyFAI's `line.split(":", 1)[1].strip()`.
/// Returns `""` if there is no colon (the call sites only reach here after a
/// `contains(":")` check, so the colon is always present).
fn after_colon(line: &str) -> String {
    line.split_once(':')
        .map(|(_, rest)| rest)
        .unwrap_or("")
        .trim()
        .to_string()
}

impl CalibrantConfig {
    /// Port of `CalibrantConfig.from_dspacing`: parse a `.D` file's text.
    ///
    /// Mirrors pyFAI's two parse modes: the structured `d # (h k l) mult int`
    /// form (`hash_pos == 1`) and the `generic` fallback where bare floats are
    /// read until a non-numeric token. The "weak" intensity convention
    /// (`has_weak_reflection`) and the parenthesis-aware Miller scan are
    /// reproduced verbatim so the reflection list matches byte-for-byte.
    pub fn from_dspacing_str(text: &str) -> CalibrantConfig {
        let mut config = CalibrantConfig::default();
        let raw: Vec<String> = text.lines().map(|l| l.trim().to_string()).collect();

        let has_weak_reflection = raw.join(" ").to_lowercase().contains("weak");

        let mut begining = true;
        let mut generic = false;

        for line in &raw {
            if begining && line.starts_with('#') {
                let line = line.trim_matches(|c| c == '#' || c == ' ' || c == '\t');
                if line.contains("Calibrant:") {
                    let name = after_colon(line);
                    if let Some(idx) = name.find('(') {
                        config.description = name[..idx].trim().to_string();
                        // Balance nested parentheses, e.g. `Vanadinite (Pb5(BO4)3Cl)`.
                        let mut cnt = 0i32;
                        let mut lname = String::new();
                        for c in name[idx..].chars() {
                            lname.push(c);
                            if c == '(' {
                                cnt += 1;
                            } else if c == ')' {
                                cnt -= 1;
                            }
                            if cnt == 0 {
                                break;
                            }
                        }
                        // Drop the outer parentheses (lname[1:-1]).
                        let inner: String = {
                            let chars: Vec<char> = lname.chars().collect();
                            if chars.len() >= 2 {
                                chars[1..chars.len() - 1].iter().collect()
                            } else {
                                String::new()
                            }
                        };
                        config.name = inner.trim().to_string();
                    } else {
                        config.name = name.trim().to_string();
                    }
                    continue;
                } else if line.contains("Ref:") {
                    config.reference = after_colon(line);
                    continue;
                } else if line.contains("Cell:") {
                    let cell = after_colon(line);
                    if cell.contains('(') && cell.contains(')') {
                        let idx = cell.find('(').unwrap();
                        let close = cell.find(')').unwrap();
                        config.space_group = cell[idx + 1..close].trim().to_string();
                        config.cell = cell[..idx].trim().to_string();
                    } else {
                        config.cell = cell;
                    }
                    continue;
                } else {
                    if config.cell.is_empty() {
                        config.cell = line.to_string();
                    }
                }
                continue;
            }
            begining = false;
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }
            if generic {
                for word in &words {
                    if word.starts_with('#') {
                        break;
                    }
                    match word.parse::<f64>() {
                        Ok(value) => config.reflections.push(Reflection::new(value)),
                        Err(_) => break,
                    }
                }
                continue;
            }
            let hash_pos = words.iter().position(|&w| w == "#");
            let hash_pos = match hash_pos {
                Some(p) => p,
                None => {
                    // No `#` marker: every word is a bare d-spacing, then switch
                    // to generic mode (pyFAI's `generic = True`).
                    for w in &words {
                        if let Ok(v) = w.parse::<f64>() {
                            config.reflections.push(Reflection::new(v));
                        }
                    }
                    generic = true;
                    continue;
                }
            };
            if hash_pos == 1 {
                if words[0].starts_with('#') {
                    continue;
                }
                let dval: f64 = match words[0].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let mut reflection = Reflection::new(dval);
                if has_weak_reflection {
                    reflection.intensity = Some(1.0);
                }
                // Locate the Miller index `(h k l)` starting after the `#`.
                let mut start_miller: Option<usize> = None;
                let mut end_miller: Option<usize> = None;
                for (i, j) in words.iter().enumerate().skip(2) {
                    if j.starts_with('(') {
                        start_miller = Some(i);
                        if j.ends_with(')') {
                            end_miller = Some(i);
                            break;
                        }
                        continue;
                    }
                    if j.ends_with(')') {
                        end_miller = Some(i);
                        break;
                    }
                }
                if let (Some(s), Some(e)) = (start_miller, end_miller) {
                    reflection.hkl = Miller::parse(&words[s..=e].join(" "));
                    if words.len() > e + 1 {
                        let mult = words[e + 1];
                        if mult.starts_with('#') {
                            config.reflections.push(reflection);
                            continue;
                        } else if mult.chars().all(|c| c.is_ascii_digit()) && !mult.is_empty() {
                            reflection.multiplicity = mult.parse::<u32>().ok();
                        }
                    }
                    if words.len() > e + 2 {
                        let intensity = words[e + 2];
                        if intensity.starts_with('#') {
                            config.reflections.push(reflection);
                            continue;
                        }
                        match intensity.parse::<f64>() {
                            Ok(value) => reflection.intensity = Some(value),
                            Err(_) => {
                                if intensity.to_lowercase().contains("weak") {
                                    reflection.intensity = Some(0.0);
                                }
                            }
                        }
                    }
                }
                config.reflections.push(reflection);
            }
        }
        config
    }

    /// The d-spacing list in file order, `[ref.dspacing for ref in reflections]`.
    pub fn dspacing(&self) -> Vec<f64> {
        self.reflections.iter().map(|r| r.dspacing).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miller_parse_basic() {
        assert_eq!(Miller::parse("(1 1 1)"), Some(Miller::new(1, 1, 1)));
        assert_eq!(Miller::parse("(0 0 10)"), Some(Miller::new(0, 0, 10)));
        assert_eq!(Miller::parse("1, 2, 3"), Some(Miller::new(1, 2, 3)));
        assert_eq!(Miller::parse("(1 2)"), None);
    }

    #[test]
    fn parse_al_header_and_first_rows() {
        let text = "# Calibrant: Aluminium (Al)\n\
                    # Cell: Cubic cell a=4.0495 b=4.0495 c=4.0495 alpha=90.000 beta=90.000 gamma=90.000 (Fm3m)\n\
                    # Ref: W. Witt\n\
                    \n\
                    # d_spacing  # (h k l)  mult intensity\n\
                      2.33797992 # (1 1 1)     8 100.0\n\
                      2.02475000 # (2 0 0)     6 47.49\n\
                      1.01237500 # (4 0 0)     6\n";
        let cfg = CalibrantConfig::from_dspacing_str(text);
        assert_eq!(cfg.name, "Al");
        assert_eq!(cfg.description, "Aluminium");
        assert_eq!(cfg.space_group, "Fm3m");
        assert_eq!(cfg.reflections.len(), 3);
        assert_eq!(cfg.reflections[0].dspacing, 2.33797992);
        assert_eq!(cfg.reflections[0].hkl, Some(Miller::new(1, 1, 1)));
        assert_eq!(cfg.reflections[0].multiplicity, Some(8));
        assert_eq!(cfg.reflections[0].intensity, Some(100.0));
        // No intensity column on the (4 0 0) row.
        assert_eq!(cfg.reflections[2].intensity, None);
        assert_eq!(cfg.reflections[2].multiplicity, Some(6));
    }

    #[test]
    fn parse_high_index_miller() {
        // AgBh has (0 0 10) etc.
        let text = "# d_spacing  # (h k l)  mult intensity\n  5.83800000 # (0 0 10)\n";
        let cfg = CalibrantConfig::from_dspacing_str(text);
        assert_eq!(cfg.reflections[0].hkl, Some(Miller::new(0, 0, 10)));
    }
}
