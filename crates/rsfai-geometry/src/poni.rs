//! PONI ("Point Of Normal Incidence") file parsing, ported from
//! `pyFAI/io/ponifile.py`. A PONI file is a sequence of `Key: value` lines with
//! `#` comments. We parse the scalar geometry plus the detector identity; the
//! detector pixel model itself is reproduced in `rsfai-detectors` (M2).
//!
//! Versions: v1 (no `poni_version`, detector implied by `SplineFile` /
//! `PixelSize1`/`PixelSize2`), v2/v2.1/v3 (`poni_version` + `Detector` +
//! `Detector_config` JSON). Key matching is case-insensitive; `Distance` and
//! `Dist` are accepted aliases.

use std::path::Path;

use crate::error::{GeometryError, Result};

/// Parsed PONI geometry. Lengths are SI (metres, radians).
#[derive(Debug, Clone, PartialEq)]
pub struct PoniFile {
    pub poni_version: Option<u32>,
    pub detector: Option<String>,
    /// Raw `Detector_config` JSON text, if present (parsed in M2).
    pub detector_config: Option<String>,
    /// Sample–detector distance `L` (m).
    pub dist: f64,
    /// PONI coordinate along the slow (Y) axis (m).
    pub poni1: f64,
    /// PONI coordinate along the fast (X) axis (m).
    pub poni2: f64,
    pub rot1: f64,
    pub rot2: f64,
    pub rot3: f64,
    /// X-ray wavelength (m), if recorded.
    pub wavelength: Option<f64>,
    /// Pixel size along slow axis (m), only present in v1-style files.
    pub pixel1: Option<f64>,
    /// Pixel size along fast axis (m), only present in v1-style files.
    pub pixel2: Option<f64>,
}

impl PoniFile {
    /// Parse PONI text.
    pub fn parse(text: &str) -> Result<Self> {
        let mut poni_version = None;
        let mut detector = None;
        let mut detector_config = None;
        let mut dist = None;
        let mut poni1 = None;
        let mut poni2 = None;
        let mut rot1 = None;
        let mut rot2 = None;
        let mut rot3 = None;
        let mut wavelength = None;
        let mut pixel1 = None;
        let mut pixel2 = None;

        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            let parse_f64 = |v: &str| -> Result<f64> {
                v.parse::<f64>()
                    .map_err(|e| GeometryError::PoniParse(format!("{key}: {e}")))
            };

            match key.to_ascii_lowercase().as_str() {
                "poni_version" => {
                    // Versions can be "2" or "2.1"; keep the major component.
                    let major = value.split('.').next().unwrap_or(value);
                    poni_version = major.parse::<u32>().ok();
                }
                "detector" => detector = Some(value.to_string()),
                "detector_config" => detector_config = Some(value.to_string()),
                "distance" | "dist" => dist = Some(parse_f64(value)?),
                "poni1" => poni1 = Some(parse_f64(value)?),
                "poni2" => poni2 = Some(parse_f64(value)?),
                "rot1" => rot1 = Some(parse_f64(value)?),
                "rot2" => rot2 = Some(parse_f64(value)?),
                "rot3" => rot3 = Some(parse_f64(value)?),
                "wavelength" => wavelength = Some(parse_f64(value)?),
                "pixelsize1" => pixel1 = Some(parse_f64(value)?),
                "pixelsize2" => pixel2 = Some(parse_f64(value)?),
                _ => {} // SplineFile, Detector-specific keys, etc.: ignored in M1.
            }
        }

        Ok(PoniFile {
            poni_version,
            detector,
            detector_config,
            dist: dist.ok_or(GeometryError::PoniMissingKey("Distance"))?,
            poni1: poni1.ok_or(GeometryError::PoniMissingKey("Poni1"))?,
            poni2: poni2.ok_or(GeometryError::PoniMissingKey("Poni2"))?,
            rot1: rot1.ok_or(GeometryError::PoniMissingKey("Rot1"))?,
            rot2: rot2.ok_or(GeometryError::PoniMissingKey("Rot2"))?,
            rot3: rot3.ok_or(GeometryError::PoniMissingKey("Rot3"))?,
            wavelength,
            pixel1,
            pixel2,
        })
    }

    /// Parse a PONI file from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p).map_err(|source| GeometryError::Io {
            path: p.to_string_lossy().into_owned(),
            source,
        })?;
        Self::parse(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PILATUS1M_PONI: &str = "\
# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis
poni_version: 2
Detector: Pilatus1M
Detector_config: {}
Distance: 1.58323111834
Poni1: 0.0334170169115
Poni2: 0.0412277798782
Rot1: 0.00648735642526
Rot2: 0.00755810191106
Rot3: 4.12987220385e-08
Wavelength: 1.0e-10
";

    #[test]
    fn parses_pilatus1m_v2() {
        let p = PoniFile::parse(PILATUS1M_PONI).unwrap();
        assert_eq!(p.poni_version, Some(2));
        assert_eq!(p.detector.as_deref(), Some("Pilatus1M"));
        assert_eq!(p.detector_config.as_deref(), Some("{}"));
        // Exact f64 round-trip of the decimal literals.
        assert_eq!(p.dist, 1.58323111834);
        assert_eq!(p.poni1, 0.0334170169115);
        assert_eq!(p.poni2, 0.0412277798782);
        assert_eq!(p.rot1, 0.00648735642526);
        assert_eq!(p.rot2, 0.00755810191106);
        assert_eq!(p.rot3, 4.12987220385e-08);
        assert_eq!(p.wavelength, Some(1.0e-10));
    }

    #[test]
    fn missing_key_errors() {
        let err = PoniFile::parse("Poni1: 0.1\nPoni2: 0.2\n").unwrap_err();
        assert!(matches!(err, GeometryError::PoniMissingKey("Distance")));
    }
}
