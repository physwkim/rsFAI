//! How the PONI parameters move the diffraction rings — the Rust analogue of
//! pyFAI's `geometry.ipynb` tutorial ("Geometries in pyFAI").
//!
//! The notebook builds a fake 1000×1000 / 100 µm detector at a 1 m distance and
//! 0.1 nm wavelength, then walks through each geometry knob — the two
//! beam-centre offsets `poni1`/`poni2`, the sample–detector distance `dist`, and
//! the three rotations `rot1`/`rot2`/`rot3` — showing how each shifts the rings
//! drawn by a silver-behenate calibrant. The headless equivalent here drives the
//! landed geometry primitive directly (`calc_pos_zyx` → `center_array`) and
//! reports the *same* parameter effects as numbers on the per-pixel maps:
//!
//!   * 2θ radial map (`center_array(TTH_DEG)`): sweeping `poni1`/`poni2` moves
//!     the map's minimum — the beam-centre projection, i.e. the ring centre —
//!     across the detector; increasing `dist` shrinks the 2θ a given pixel
//!     subtends (the rings contract toward the centre).
//!   * `rot1`/`rot2` tilt the detector: the 2θ minimum shifts off the geometric
//!     PONI pixel and the span becomes asymmetric (the rings turn elliptical).
//!   * chi azimuthal map (`center_array(CHI_DEG)`): the notebook's closing point
//!     is that `rot3` rotates the azimuthal frame about the beam ("increasing
//!     rot3 creates more negative azimuthal angles"). The chi value at a fixed
//!     off-centre pixel shifts by ≈ −rot3, with the radial map left invariant.
//!
//! Runs fully offline: the geometry is constructed in code (no `.poni`), nothing
//! is read or plotted. We use the `rsfai-geometry` primitive that this crate
//! owns — the pixel→lab transform and the unit equations — rather than the
//! top-level `AzimuthalIntegrator` (a higher crate this one does not depend on),
//! so the demo stays within the geometry crate's own API. pyFAI's
//! `fake_calibration_image`, the custom `ShiftedDetector`, and matplotlib display
//! are out of remit; the point is the parameter→ring-position mapping, shown
//! numerically. The bit-exact parity check against pyFAI lives in the golden
//! tests, not here.
//!
//!   cargo run --release --example geometry_params -p rsfai-geometry

use rsfai_detectors::Detector;
use rsfai_geometry::{calc_pos_zyx, center_array, PosZyx, Unit};

/// The notebook's fake detector: 1000×1000 pixels of 100 µm, gapless, at 1 m,
/// with a 0.1 nm beam. `Detector::generic` gives orientation 0 — the closest
/// landed analogue of the tutorial's bare `Detector`.
const PIXEL: f64 = 100e-6;
const SHAPE: (usize, usize) = (1000, 1000);
const DIST: f64 = 1.0;
const WAVELENGTH: f64 = 1e-10;

/// One geometry setting: the PONI scalars the notebook sweeps. Distances are
/// metres, rotations radians — matching pyFAI's `AzimuthalIntegrator`.
struct Geometry {
    poni1: f64,
    poni2: f64,
    dist: f64,
    rot1: f64,
    rot2: f64,
    rot3: f64,
}

impl Geometry {
    /// Lab coordinates `(z, y, x)` for every pixel centre: the generic detector's
    /// pixel grid fed through the PONI rotation `calc_pos_zyx` (orientation 0,
    /// flat ⇒ `p3 = None`). This is exactly what `AzimuthalIntegrator::pixel_positions`
    /// does, built here from the geometry primitive.
    fn pixel_positions(&self) -> PosZyx {
        let det = Detector::generic(PIXEL, PIXEL, SHAPE);
        let (p1, p2) = det.centers_f64();
        calc_pos_zyx(
            self.dist,
            self.poni1,
            self.poni2,
            self.rot1,
            self.rot2,
            self.rot3,
            &p1,
            &p2,
            None,
            det.orientation,
        )
    }

    /// The scaled per-pixel center map for `unit` (`center_array`, the radial /
    /// azimuthal value pyFAI reports per pixel).
    fn map(&self, unit: Unit) -> Vec<f64> {
        let pos = self.pixel_positions();
        center_array(unit, &pos.x, &pos.y, &pos.z, WAVELENGTH)
    }
}

/// `(min, argmin, max)` of a flat array — locates the radial-map minimum (the
/// beam-centre projection / ring centre) and its full extent.
fn min_argmin_max(a: &[f64]) -> (f64, usize, f64) {
    let mut lo = f64::INFINITY;
    let mut lo_i = 0usize;
    let mut hi = f64::NEG_INFINITY;
    for (i, &v) in a.iter().enumerate() {
        if v < lo {
            lo = v;
            lo_i = i;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, lo_i, hi)
}

/// Report a geometry's 2θ radial map: where its minimum sits (the ring centre,
/// in pixel row/col) and the full 2θ span across the detector.
fn report_radial_map(label: &str, g: &Geometry) {
    let tth = g.map(Unit::TTH_DEG);
    let (lo, lo_i, hi) = min_argmin_max(&tth);
    let (row, col) = (lo_i / SHAPE.1, lo_i % SHAPE.1);
    println!("{label}");
    println!(
        "  poni1 = {:.5} m  poni2 = {:.5} m  dist = {:.3} m  rot1/2/3 = {:.3}/{:.3}/{:.3} rad",
        g.poni1, g.poni2, g.dist, g.rot1, g.rot2, g.rot3
    );
    println!(
        "  ring centre (2th min) at pixel (row {row}, col {col})   2th span [{lo:.4}, {hi:.4}] deg"
    );
}

fn main() {
    println!(
        "Geometry sweep: {}x{} px of {:.0} um, dist = {:.1} m, lambda = {:.3} nm, unit = 2th_deg\n",
        SHAPE.0,
        SHAPE.1,
        PIXEL * 1e6,
        DIST,
        WAVELENGTH * 1e9
    );

    // The notebook moves the beam centre from the detector origin to the middle:
    // poni = pixel * N / 2.
    let centre = PIXEL * SHAPE.0 as f64 / 2.0;
    let centred = |dist, rot1, rot2, rot3| Geometry {
        poni1: centre,
        poni2: centre,
        dist,
        rot1,
        rot2,
        rot3,
    };

    // ---- poni1 / poni2: translate the ring centre across the detector ----
    println!("== Translation orthogonal to the beam (poni1, poni2) ==");
    report_radial_map(
        "poni = (0, 0): beam at the detector origin",
        &Geometry {
            poni1: 0.0,
            poni2: 0.0,
            dist: DIST,
            rot1: 0.0,
            rot2: 0.0,
            rot3: 0.0,
        },
    );
    report_radial_map(
        "poni1 set to the vertical centre",
        &Geometry {
            poni1: centre,
            poni2: 0.0,
            dist: DIST,
            rot1: 0.0,
            rot2: 0.0,
            rot3: 0.0,
        },
    );
    report_radial_map(
        "poni1 + poni2 set to the detector centre",
        &centred(DIST, 0.0, 0.0, 0.0),
    );

    // ---- dist: increasing the distance contracts the rings ----
    // Same beam centre, three distances. A farther detector means each pixel
    // subtends a smaller 2θ, so the edge 2θ (the span maximum) drops.
    println!("\n== Sample-detector distance (dist) ==");
    for dist in [0.5, 1.0, 1.5] {
        report_radial_map(
            &format!("dist = {dist:.1} m"),
            &centred(dist, 0.0, 0.0, 0.0),
        );
    }

    // ---- rot1 / rot2: tilt the detector, ellipsing the rings ----
    // A rotation about an in-plane axis tilts the detector: the ring centre's 2θ
    // minimum shifts off the geometric PONI pixel and the span turns asymmetric.
    println!("\n== Rotations rot1, rot2 (detector tilt) ==");
    report_radial_map("no rotation (reference)", &centred(DIST, 0.0, 0.0, 0.0));
    report_radial_map("rot1 = 0.2 rad", &centred(DIST, 0.2, 0.0, 0.0));
    report_radial_map("rot2 = 0.2 rad", &centred(DIST, 0.0, 0.2, 0.0));

    // ---- rot3: rotates the azimuthal frame about the beam ----
    // The notebook's point: rot3 has "no visible effect on the image" (the radial
    // map is invariant) and its effect shows only in the azimuthal angle —
    // "increasing rot3 creates more negative azimuthal angles". We read the chi
    // value at one fixed off-centre pixel (the detector corner) for rot3 = 0 vs
    // 0.2 rad, and confirm the 2θ map is unchanged there.
    println!("\n== Rotation rot3 (azimuthal rotation about the beam) ==");
    let probe = 0usize; // pixel (row 0, col 0): off the centred beam ⇒ a defined chi.
    for rot3 in [0.0, 0.2] {
        let g = centred(DIST, 0.0, 0.0, rot3);
        let chi = g.map(Unit::CHI_DEG);
        let tth = g.map(Unit::TTH_DEG);
        println!(
            "rot3 = {rot3:.1} rad -> at pixel (row 0, col 0): chi = {:8.4} deg   2th = {:.4} deg",
            chi[probe], tth[probe]
        );
    }
}
