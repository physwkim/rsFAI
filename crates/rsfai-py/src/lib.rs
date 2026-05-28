//! `rsfai` — PyO3 bindings over the bit-exact rsFAI integration kernels.
//!
//! This module is a thin, numpy-in/numpy-out shim around the engines in
//! `rsfai-preproc` and `rsfai-integrate`. It exists for **in-process
//! side-by-side validation**: a test running in the same Python interpreter as
//! `pyFAI` feeds both libraries the identical input arrays and compares the
//! outputs bit-for-bit (`numpy.ndarray.tobytes()` / `.view` equality).
//!
//! The bindings add no arithmetic of their own — every value is produced by the
//! already-validated Rust engines. The only conversions here are:
//!   * borrowing C-contiguous numpy buffers as Rust slices (zero-copy in),
//!   * mapping the integer `error_model` code (0/1/2/3/4) to [`ErrorModel`],
//!   * copying the engine output `Vec`s into fresh numpy arrays (out).
//!
//! Preprocessed rows (`prep`) are passed as an `(npix, 4)` f32 array — its
//! C-order flattening is exactly the `[signal, variance, norm, count]`-per-pixel
//! layout the engines consume. Corner arrays are passed pre-flattened to f64
//! (`(npix*4*2,)`), matching the engines' `(npix, 4, 2)` C-order contract.
//!
//! Alongside the per-kernel functions, the module exposes the high-level
//! [`AzimuthalIntegrator`](PyAzimuthalIntegrator): `load(poni)` then
//! `integrate1d`/`integrate2d` of a detector frame — PONI + image in, nothing
//! else — wrapping `rsfai_engine::AzimuthalIntegrator` so an in-process test can
//! drive it the same way it drives `pyFAI.load(poni).integrate1d_ng(...)`. It
//! adds no arithmetic either; it only marshals numpy in / dict out and maps the
//! radial-unit string to the engine enum.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rsfai_core::dtype::ErrorModel;
use rsfai_engine::{
    Algo, AzimuthalIntegrator as RsAzimuthalIntegrator, Integrate1dResult, Integrate2dResult,
    IntegrationOptions, Method, RadialUnit, Split,
};
use rsfai_integrate::{
    build_bbox_csc_1d as rs_build_bbox_csc_1d, build_bbox_csc_2d as rs_build_bbox_csc_2d,
    build_bbox_csr_1d as rs_build_bbox_csr_1d, build_bbox_csr_2d as rs_build_bbox_csr_2d,
    build_bbox_lut_1d as rs_build_bbox_lut_1d, build_bbox_lut_2d as rs_build_bbox_lut_2d,
    build_full_csc_1d as rs_build_full_csc_1d, build_full_csc_2d as rs_build_full_csc_2d,
    build_full_csr_1d as rs_build_full_csr_1d, build_full_csr_2d as rs_build_full_csr_2d,
    build_full_lut_1d as rs_build_full_lut_1d, build_full_lut_2d as rs_build_full_lut_2d,
    csc_integrate1d as rs_csc_integrate1d, csc_integrate2d as rs_csc_integrate2d,
    csr_integrate1d as rs_csr_integrate1d, csr_integrate2d as rs_csr_integrate2d,
    histogram1d as rs_histogram1d, histogram1d_bbox as rs_histogram1d_bbox,
    histogram1d_full as rs_histogram1d_full, histogram2d as rs_histogram2d,
    histogram2d_bbox as rs_histogram2d_bbox, histogram2d_full as rs_histogram2d_full,
    histogram2d_pseudo as rs_histogram2d_pseudo, histogram_preproc as rs_histogram_preproc,
    lut_integrate1d as rs_lut_integrate1d, lut_integrate2d as rs_lut_integrate2d, Bbox2dBounds,
    Csc, Csr, CsrIntegrate1d, Hist2dOptions, Integrate1d, Integrate2d, Lut,
};
use rsfai_preproc::{preproc4 as rs_preproc4, PreprocOptions};

/// A built 1D CSR returned to Python: `(data, indices, indptr, bin_centers)`.
type Csr1dPy<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
);

/// `histogram_preproc` return: `(prop (npt, 5), position (npt,))`.
type HistoPreprocPy<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

/// A built 2D CSR: `(data, indices, indptr, bin_centers0, bin_centers1)`.
type Csr2dPy<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// A built 1D dense LUT: `(idx, coef, lut_size, bin_centers)`. `idx`/`coef` are
/// the flattened `(n_bins, lut_size)` row-major matrix.
type Lut1dPy<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    usize,
    Bound<'py, PyArray1<f64>>,
);

/// A built 2D dense LUT: `(idx, coef, lut_size, bin_centers0, bin_centers1)`.
type Lut2dPy<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    usize,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Map pyFAI's integer error-model code to the Rust enum.
fn error_model(code: i32) -> PyResult<ErrorModel> {
    ErrorModel::from_code(code).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown error_model code {code} (expected 0=no, 1=variance, 2=poisson, \
             3=azimuthal, 4=hybrid)"
        ))
    })
}

/// Map a pyFAI radial-unit string (the names used in the golden manifests and
/// passed to `ai.integrate*`) to the engine [`RadialUnit`].
fn radial_unit(name: &str) -> PyResult<RadialUnit> {
    match name {
        "q_nm^-1" => Ok(RadialUnit::Q_NM_INV),
        "q_A^-1" => Ok(RadialUnit::Q_A_INV),
        "2th_deg" => Ok(RadialUnit::TTH_DEG),
        "2th_rad" => Ok(RadialUnit::TTH_RAD),
        "r_mm" => Ok(RadialUnit::R_MM),
        "r_m" => Ok(RadialUnit::R_M),
        other => Err(PyValueError::new_err(format!(
            "unsupported radial unit {other:?} \
             (expected one of q_nm^-1, q_A^-1, 2th_deg, 2th_rad, r_mm, r_m)"
        ))),
    }
}

/// Map a pyFAI method tuple to the engine [`Method`]. `None` ⇒ the default
/// `("no", "histogram")`. The tuple's first two elements are `(split, algo)`;
/// the third (the implementation, e.g. `"cython"`) is ignored — the port is the
/// cython algorithm. `"pseudo"` (2D-only, not ported) and any unknown token are
/// errors so an unsupported method never silently runs a different path.
fn parse_method(method: Option<&[String]>) -> PyResult<Method> {
    let Some(m) = method else {
        return Ok(Method::default());
    };
    if m.len() < 2 {
        return Err(PyValueError::new_err(
            "method must be a (split, algo[, impl]) tuple of strings",
        ));
    }
    let split = match m[0].as_str() {
        "no" => Split::No,
        "bbox" => Split::Bbox,
        "full" => Split::Full,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported split {other:?} (expected no, bbox, or full)"
            )))
        }
    };
    let algo = match m[1].as_str() {
        "histogram" => Algo::Histogram,
        "csr" => Algo::Csr,
        "lut" => Algo::Lut,
        "csc" => Algo::Csc,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported algo {other:?} (expected histogram, csr, lut, or csc)"
            )))
        }
    };
    Ok(Method { split, algo })
}

/// Borrow a contiguous 1D readonly numpy array as a slice, or a clear error.
fn as_slice_1d<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|e| PyValueError::new_err(format!("array must be C-contiguous: {e}")))
}

/// Borrow an `(n, 4)` readonly f32 array as its flat C-order slice (length `4n`).
fn as_slice_2d<'a>(a: &'a PyReadonlyArray2<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|e| PyValueError::new_err(format!("array must be C-contiguous: {e}")))
}

/// Optional contiguous i8 mask as a slice.
fn mask_slice<'a>(mask: &'a Option<PyReadonlyArray1<'a, i8>>) -> PyResult<Option<&'a [i8]>> {
    match mask {
        Some(m) => Ok(Some(as_slice_1d(m)?)),
        None => Ok(None),
    }
}

// --------------------------------------------------------------------------
// Preproc
// --------------------------------------------------------------------------

/// Per-pixel preprocessing (`preproc(..., split_result=4)`), returned as an
/// `(npix, 4)` f32 array of `[signal, variance, normalization, count]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    data, *, dark=None, flat=None, solidangle=None, polarization=None,
    absorption=None, mask=None, variance=None, dark_variance=None,
    normalization_factor=1.0, poissonian=false, check_dummy=false,
    dummy=0.0, delta_dummy=0.0, apply_normalization=true,
))]
fn preproc4<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    dark: Option<PyReadonlyArray1<'py, f32>>,
    flat: Option<PyReadonlyArray1<'py, f32>>,
    solidangle: Option<PyReadonlyArray1<'py, f32>>,
    polarization: Option<PyReadonlyArray1<'py, f32>>,
    absorption: Option<PyReadonlyArray1<'py, f32>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    variance: Option<PyReadonlyArray1<'py, f32>>,
    dark_variance: Option<PyReadonlyArray1<'py, f32>>,
    normalization_factor: f32,
    poissonian: bool,
    check_dummy: bool,
    dummy: f32,
    delta_dummy: f32,
    apply_normalization: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data_s = as_slice_1d(&data)?;
    let opt = PreprocOptions {
        dark: opt_slice(&dark)?,
        flat: opt_slice(&flat)?,
        solidangle: opt_slice(&solidangle)?,
        polarization: opt_slice(&polarization)?,
        absorption: opt_slice(&absorption)?,
        mask: mask_slice(&mask)?,
        variance: opt_slice(&variance)?,
        dark_variance: opt_slice(&dark_variance)?,
        normalization_factor,
        poissonian,
        check_dummy,
        dummy,
        delta_dummy,
        apply_normalization,
    };
    let flat_out = rs_preproc4(data_s, &opt);
    let npix = data_s.len();
    debug_assert_eq!(flat_out.len(), 4 * npix);
    let arr = Array2::from_shape_vec((npix, 4), flat_out)
        .map_err(|e| PyValueError::new_err(format!("preproc reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Optional contiguous f32 array as a slice.
fn opt_slice<'a>(a: &'a Option<PyReadonlyArray1<'a, f32>>) -> PyResult<Option<&'a [f32]>> {
    match a {
        Some(x) => Ok(Some(as_slice_1d(x)?)),
        None => Ok(None),
    }
}

// --------------------------------------------------------------------------
// Histogram (no split)
// --------------------------------------------------------------------------

/// `histogram_preproc`: bin a preprocessed `(npix, 4)` array into `npt` bins,
/// returning `(prop, position)` where `prop` is the `(npt, 5)` f64 accumulator
/// `[signal, variance, normalization, count, norm^2]` and `position` the f64 bin
/// centers — matching `pyFAI.ext.histogram.histogram_preproc`.
#[pyfunction]
#[pyo3(signature = (radial, prep, npt, *, bin_range=None, error_model=0))]
fn histogram_preproc<'py>(
    py: Python<'py>,
    radial: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    npt: usize,
    bin_range: Option<(f64, f64)>,
    error_model: i32,
) -> PyResult<HistoPreprocPy<'py>> {
    let radial_s = as_slice_1d(&radial)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let (prop, position) = rs_histogram_preproc(radial_s, prep_s, npt, bin_range, em);
    let mut flat = Vec::with_capacity(prop.len() * 5);
    for row in &prop {
        flat.extend_from_slice(row);
    }
    let prop_arr = Array2::from_shape_vec((prop.len(), 5), flat)
        .map_err(|e| PyValueError::new_err(format!("prop reshape failed: {e}")))?;
    Ok((prop_arr.into_pyarray(py), position.into_pyarray(py)))
}

/// `histogram1d`: full 1D no-split integration (bin + reduce). Returns a dict of
/// the `Integrate1dtpl` fields.
#[pyfunction]
#[pyo3(signature = (radial, prep, npt, *, bin_range=None, error_model=0, empty=0.0))]
fn histogram1d<'py>(
    py: Python<'py>,
    radial: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    npt: usize,
    bin_range: Option<(f64, f64)>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let radial_s = as_slice_1d(&radial)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let r = rs_histogram1d(radial_s, prep_s, npt, bin_range, em, empty);
    integrate1d_to_dict(py, &r)
}

/// `histogram2d`: full 2D no-split integration. Returns a dict of the
/// `Integrate2dtpl` fields; the 2D arrays are flat in (azimuthal, radial)
/// C-order — reshape to `(bins_azim, bins_rad)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    radial, azimuthal, prep, *, bins, mask=None, radial_range=None,
    azimuth_range=None, error_model=0, allow_radial_neg=false,
    chi_disc_at_pi=true, pos1_period=0.0, empty=0.0,
))]
fn histogram2d<'py>(
    py: Python<'py>,
    radial: PyReadonlyArray1<'py, f64>,
    azimuthal: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    bins: (usize, usize),
    mask: Option<PyReadonlyArray1<'py, i8>>,
    radial_range: Option<(f64, f64)>,
    azimuth_range: Option<(f64, f64)>,
    error_model: i32,
    allow_radial_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let radial_s = as_slice_1d(&radial)?;
    let azim_s = as_slice_1d(&azimuthal)?;
    let prep_s = as_slice_2d(&prep)?;
    let opts = Hist2dOptions {
        bins,
        radial_range,
        azimuth_range,
        error_model: self::error_model(error_model)?,
        allow_radial_neg,
        chi_disc_at_pi,
        pos1_period,
        empty,
    };
    let r = rs_histogram2d(radial_s, azim_s, prep_s, mask_slice(&mask)?, &opts);
    integrate2d_to_dict(py, &r)
}

/// `histogram1d_bbox`: 1D direct-split bbox histogram (`histoBBox1d_engine`).
/// `pos0`/`delta_pos0` are the unscaled radial center / half-width per pixel.
/// Returns a dict of the `Integrate1dtpl` fields (f64 binned sums, like the CSR
/// path).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (pos0, delta_pos0, prep, *, mask=None, npt, error_model=0, empty=0.0, allow_pos0_neg=false))]
fn histogram1d_bbox<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    delta_pos0: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    npt: usize,
    error_model: i32,
    empty: f32,
    allow_pos0_neg: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let pos0_s = as_slice_1d(&pos0)?;
    let delta_s = as_slice_1d(&delta_pos0)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let r = rs_histogram1d_bbox(
        pos0_s,
        delta_s,
        prep_s,
        mask_slice(&mask)?,
        npt,
        em,
        empty,
        allow_pos0_neg,
        None,
        None,
    );
    csr_integrate1d_to_dict(py, &r)
}

/// `histogram2d_bbox`: 2D direct-split bbox histogram (`histoBBox2d_engine`).
/// `pos0`/`delta_pos0` are the unscaled radial center / half-width; `pos1`/
/// `delta_pos1` the radian azimuthal (chi) center / half-width. Returns a dict
/// of the `Integrate2dtpl` fields; the 2D arrays are flat in (azimuthal, radial)
/// C-order — reshape to `(bins_azim, bins_rad)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    pos0, delta_pos0, pos1, delta_pos1, prep, *, bins, mask=None,
    allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period, error_model=0, empty=0.0,
))]
fn histogram2d_bbox<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    delta_pos0: PyReadonlyArray1<'py, f64>,
    pos1: PyReadonlyArray1<'py, f64>,
    delta_pos1: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    bins: (usize, usize),
    mask: Option<PyReadonlyArray1<'py, i8>>,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let pos0_s = as_slice_1d(&pos0)?;
    let delta0_s = as_slice_1d(&delta_pos0)?;
    let pos1_s = as_slice_1d(&pos1)?;
    let delta1_s = as_slice_1d(&delta_pos1)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let r = rs_histogram2d_bbox(
        pos0_s,
        delta0_s,
        pos1_s,
        delta1_s,
        prep_s,
        mask_slice(&mask)?,
        bins,
        &bounds,
        em,
        empty,
    );
    integrate2d_to_dict(py, &r)
}

/// `histogram1d_full`: 1D full pixel-splitting histogram (`fullSplit1D_engine`).
/// `corners` is the `(npix, 4, 2)` array pre-flattened to f64 (length `8*npix`).
/// Returns a dict of the `Integrate1dtpl` fields (f64 binned sums, like the CSR
/// path).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    corners, prep, *, mask=None, npt, error_model=0, empty=0.0,
    allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period,
))]
fn histogram1d_full<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    npt: usize,
    error_model: i32,
    empty: f32,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let corners_s = as_slice_1d(&corners)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let r = rs_histogram1d_full(
        corners_s,
        prep_s,
        mask_slice(&mask)?,
        npt,
        em,
        empty,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        None,
        None,
    );
    csr_integrate1d_to_dict(py, &r)
}

/// `histogram2d_full`: 2D full pixel-splitting histogram (`fullSplit2D_engine`).
/// `corners` is the `(npix, 4, 2)` array pre-flattened to f64. Returns a dict of
/// the `Integrate2dtpl` fields; the 2D arrays are flat in (azimuthal, radial)
/// C-order — reshape to `(bins_azim, bins_rad)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    corners, prep, *, bins, mask=None,
    allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period, error_model=0, empty=0.0,
))]
fn histogram2d_full<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    bins: (usize, usize),
    mask: Option<PyReadonlyArray1<'py, i8>>,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let corners_s = as_slice_1d(&corners)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let r = rs_histogram2d_full(
        corners_s,
        prep_s,
        mask_slice(&mask)?,
        bins,
        &bounds,
        em,
        empty,
    );
    integrate2d_to_dict(py, &r)
}

/// `histogram2d_pseudo`: 2D pseudo pixel-splitting histogram
/// (`pseudoSplit2D_engine`, 2D only). `corners` is the `(npix, 4, 2)` array
/// pre-flattened to f64. The engine forwards no `pos1_period` (boundaries use
/// `calc_boundaries` with `clip_pos1=False`), so unlike `histogram2d_full` this
/// takes no `pos1_period`. Returns a dict of the `Integrate2dtpl` fields; the 2D
/// arrays are flat in (azimuthal, radial) C-order — reshape to `(bins_azim,
/// bins_rad)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    corners, prep, *, bins, mask=None,
    allow_pos0_neg=false, chi_disc_at_pi=true, error_model=0, empty=0.0,
))]
fn histogram2d_pseudo<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    prep: PyReadonlyArray2<'py, f32>,
    bins: (usize, usize),
    mask: Option<PyReadonlyArray1<'py, i8>>,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let corners_s = as_slice_1d(&corners)?;
    let prep_s = as_slice_2d(&prep)?;
    let em = self::error_model(error_model)?;
    let r = rs_histogram2d_pseudo(
        corners_s,
        prep_s,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        em,
        empty,
    );
    integrate2d_to_dict(py, &r)
}

// --------------------------------------------------------------------------
// CSR build
// --------------------------------------------------------------------------

/// `build_bbox_csr_1d`: returns `(data, indices, indptr, bin_centers)`.
#[pyfunction]
#[pyo3(signature = (pos0, *, delta_pos0=None, mask=None, bins, allow_pos0_neg=false))]
fn build_bbox_csr_1d<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    delta_pos0: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: usize,
    allow_pos0_neg: bool,
) -> PyResult<Csr1dPy<'py>> {
    let pos0_s = as_slice_1d(&pos0)?;
    let delta_s = match &delta_pos0 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    let (csr, centers) = rs_build_bbox_csr_1d(
        pos0_s,
        delta_s,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        None,
        None,
    );
    Ok(csr_1d_tuple(py, csr, centers))
}

/// `build_bbox_csr_2d`: returns `(data, indices, indptr, bin_centers0, bin_centers1)`.
/// `delta_pos0`/`delta_pos1` are both given (bbox split) or both omitted
/// (`("no", "csr", …)` no-split: each pixel collapses to one coef-1.0 entry).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    pos0, pos1, *, delta_pos0=None, delta_pos1=None, mask=None, bins,
    allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period,
))]
fn build_bbox_csr_2d<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    pos1: PyReadonlyArray1<'py, f64>,
    delta_pos0: Option<PyReadonlyArray1<'py, f64>>,
    delta_pos1: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Csr2dPy<'py>> {
    let d0 = match &delta_pos0 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    let d1 = match &delta_pos1 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    if d0.is_some() != d1.is_some() {
        return Err(PyValueError::new_err(
            "delta_pos0 and delta_pos1 must both be given (bbox split) or both omitted (no-split)",
        ));
    }
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let (csr, bc0, bc1) = rs_build_bbox_csr_2d(
        as_slice_1d(&pos0)?,
        d0,
        as_slice_1d(&pos1)?,
        d1,
        mask_slice(&mask)?,
        bins,
        &bounds,
    );
    Ok(csr_2d_tuple(py, csr, bc0, bc1))
}

/// `build_full_csr_1d`: `corners` is the `(npix, 4, 2)` array pre-flattened to
/// f64 (length `8*npix`). Returns `(data, indices, indptr, bin_centers)`.
#[pyfunction]
#[pyo3(signature = (corners, *, mask=None, bins, allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period))]
fn build_full_csr_1d<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: usize,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Csr1dPy<'py>> {
    let (csr, centers) = rs_build_full_csr_1d(
        as_slice_1d(&corners)?,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        None,
        None,
    );
    Ok(csr_1d_tuple(py, csr, centers))
}

/// `build_full_csr_2d`: `corners` pre-flattened to f64 (length `8*npix`).
/// Returns `(data, indices, indptr, bin_centers0, bin_centers1)`.
#[pyfunction]
#[pyo3(signature = (corners, *, mask=None, bins, allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period))]
fn build_full_csr_2d<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Csr2dPy<'py>> {
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let (csr, bc0, bc1) =
        rs_build_full_csr_2d(as_slice_1d(&corners)?, mask_slice(&mask)?, bins, &bounds);
    Ok(csr_2d_tuple(py, csr, bc0, bc1))
}

// --------------------------------------------------------------------------
// CSR apply
// --------------------------------------------------------------------------

/// Reassemble a [`Csr`] from numpy `data`/`indices`/`indptr`.
fn csr_from_parts(
    data: &PyReadonlyArray1<'_, f32>,
    indices: &PyReadonlyArray1<'_, i32>,
    indptr: &PyReadonlyArray1<'_, i32>,
) -> PyResult<Csr> {
    Ok(Csr {
        data: as_slice_1d(data)?.to_vec(),
        indices: as_slice_1d(indices)?.to_vec(),
        indptr: as_slice_1d(indptr)?.to_vec(),
    })
}

/// `csr_integrate1d`: apply a 1D CSR matrix to a preprocessed `(npix, 4)` array.
/// Returns a dict of the `Integrate1dtpl` fields.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (data, indices, indptr, prep, bin_centers, *, error_model=0, empty=0.0))]
fn csr_integrate1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    prep: PyReadonlyArray2<'py, f32>,
    bin_centers: PyReadonlyArray1<'py, f64>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let csr = csr_from_parts(&data, &indices, &indptr)?;
    let prep_s = as_slice_2d(&prep)?;
    let centers = as_slice_1d(&bin_centers)?.to_vec();
    let em = self::error_model(error_model)?;
    let r = rs_csr_integrate1d(&csr, prep_s, centers, em, empty);
    csr_integrate1d_to_dict(py, &r)
}

/// `csr_integrate2d`: apply a 2D CSR matrix. Returns a dict of the
/// `Integrate2dtpl` fields (flat (azimuthal, radial) C-order).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (data, indices, indptr, prep, bin_centers0, bin_centers1, *, error_model=0, empty=0.0))]
fn csr_integrate2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    prep: PyReadonlyArray2<'py, f32>,
    bin_centers0: PyReadonlyArray1<'py, f64>,
    bin_centers1: PyReadonlyArray1<'py, f64>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let csr = csr_from_parts(&data, &indices, &indptr)?;
    let prep_s = as_slice_2d(&prep)?;
    let bc0 = as_slice_1d(&bin_centers0)?.to_vec();
    let bc1 = as_slice_1d(&bin_centers1)?.to_vec();
    let em = self::error_model(error_model)?;
    let r = rs_csr_integrate2d(&csr, prep_s, bc0, bc1, em, empty);
    integrate2d_to_dict(py, &r)
}

// --------------------------------------------------------------------------
// CSC build + apply
// --------------------------------------------------------------------------
// The CSC matrix is the CSR LUT transposed (scipy `tocsc`): `data` are the same
// f32 coefficients permuted into column (pixel) major order, `indices` are bin
// (row) indices, `indptr` is per-PIXEL (length `n_pixels + 1`). The build tuples
// reuse the `(data, indices, indptr, centers…)` shape; the apply scatters
// pixel-major.

/// `build_bbox_csc_1d`: returns `(data, indices, indptr, bin_centers)`.
/// `delta_pos0 = None` is the `("no", "csc", …)` no-split case.
#[pyfunction]
#[pyo3(signature = (pos0, *, delta_pos0=None, mask=None, bins, allow_pos0_neg=false))]
fn build_bbox_csc_1d<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    delta_pos0: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: usize,
    allow_pos0_neg: bool,
) -> PyResult<Csr1dPy<'py>> {
    let pos0_s = as_slice_1d(&pos0)?;
    let delta_s = match &delta_pos0 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    let (csc, centers) = rs_build_bbox_csc_1d(
        pos0_s,
        delta_s,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        None,
        None,
    );
    Ok(csc_1d_tuple(py, csc, centers))
}

/// `build_bbox_csc_2d`: returns `(data, indices, indptr, bin_centers0, bin_centers1)`.
/// `delta_pos0`/`delta_pos1` both given (bbox split) or both omitted (no-split).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    pos0, pos1, *, delta_pos0=None, delta_pos1=None, mask=None, bins,
    allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period,
))]
fn build_bbox_csc_2d<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    pos1: PyReadonlyArray1<'py, f64>,
    delta_pos0: Option<PyReadonlyArray1<'py, f64>>,
    delta_pos1: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Csr2dPy<'py>> {
    let d0 = match &delta_pos0 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    let d1 = match &delta_pos1 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    if d0.is_some() != d1.is_some() {
        return Err(PyValueError::new_err(
            "delta_pos0 and delta_pos1 must both be given (bbox split) or both omitted (no-split)",
        ));
    }
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let (csc, bc0, bc1) = rs_build_bbox_csc_2d(
        as_slice_1d(&pos0)?,
        d0,
        as_slice_1d(&pos1)?,
        d1,
        mask_slice(&mask)?,
        bins,
        &bounds,
    );
    Ok(csc_2d_tuple(py, csc, bc0, bc1))
}

/// `build_full_csc_1d`: `corners` pre-flattened to f64 (length `8*npix`).
/// Returns `(data, indices, indptr, bin_centers)`.
#[pyfunction]
#[pyo3(signature = (corners, *, mask=None, bins, allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period))]
fn build_full_csc_1d<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: usize,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Csr1dPy<'py>> {
    let (csc, centers) = rs_build_full_csc_1d(
        as_slice_1d(&corners)?,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        None,
        None,
    );
    Ok(csc_1d_tuple(py, csc, centers))
}

/// `build_full_csc_2d`: `corners` pre-flattened to f64 (length `8*npix`).
/// Returns `(data, indices, indptr, bin_centers0, bin_centers1)`.
#[pyfunction]
#[pyo3(signature = (corners, *, mask=None, bins, allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period))]
fn build_full_csc_2d<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Csr2dPy<'py>> {
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let (csc, bc0, bc1) =
        rs_build_full_csc_2d(as_slice_1d(&corners)?, mask_slice(&mask)?, bins, &bounds);
    Ok(csc_2d_tuple(py, csc, bc0, bc1))
}

/// Reassemble a [`Csc`] from numpy `data`/`indices`/`indptr`.
fn csc_from_parts(
    data: &PyReadonlyArray1<'_, f32>,
    indices: &PyReadonlyArray1<'_, i32>,
    indptr: &PyReadonlyArray1<'_, i32>,
) -> PyResult<Csc> {
    Ok(Csc {
        data: as_slice_1d(data)?.to_vec(),
        indices: as_slice_1d(indices)?.to_vec(),
        indptr: as_slice_1d(indptr)?.to_vec(),
    })
}

/// `csc_integrate1d`: apply a 1D CSC matrix to a preprocessed `(npix, 4)` array.
/// Returns a dict of the `Integrate1dtpl` fields.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (data, indices, indptr, prep, bin_centers, *, error_model=0, empty=0.0))]
fn csc_integrate1d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    prep: PyReadonlyArray2<'py, f32>,
    bin_centers: PyReadonlyArray1<'py, f64>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let csc = csc_from_parts(&data, &indices, &indptr)?;
    let prep_s = as_slice_2d(&prep)?;
    let centers = as_slice_1d(&bin_centers)?.to_vec();
    let em = self::error_model(error_model)?;
    let r = rs_csc_integrate1d(&csc, prep_s, centers, em, empty);
    csr_integrate1d_to_dict(py, &r)
}

/// `csc_integrate2d`: apply a 2D CSC matrix. Returns a dict of the
/// `Integrate2dtpl` fields (flat (azimuthal, radial) C-order).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (data, indices, indptr, prep, bin_centers0, bin_centers1, *, error_model=0, empty=0.0))]
fn csc_integrate2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    indices: PyReadonlyArray1<'py, i32>,
    indptr: PyReadonlyArray1<'py, i32>,
    prep: PyReadonlyArray2<'py, f32>,
    bin_centers0: PyReadonlyArray1<'py, f64>,
    bin_centers1: PyReadonlyArray1<'py, f64>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let csc = csc_from_parts(&data, &indices, &indptr)?;
    let prep_s = as_slice_2d(&prep)?;
    let bc0 = as_slice_1d(&bin_centers0)?.to_vec();
    let bc1 = as_slice_1d(&bin_centers1)?.to_vec();
    let em = self::error_model(error_model)?;
    let r = rs_csc_integrate2d(&csc, prep_s, bc0, bc1, em, empty);
    integrate2d_to_dict(py, &r)
}

// --------------------------------------------------------------------------
// LUT build + apply
// --------------------------------------------------------------------------
// The LUT is the CSR matrix densified (`to_lut`): a flattened `(n_bins, lut_size)`
// row-major matrix of `{idx, coef}`, each bin's real entries in the leading
// columns (CSR order) and the rest zero-padding (`idx=0, coef=0.0`). The build
// tuples return `(idx, coef, lut_size, centers…)`; the apply gathers per bin,
// skipping padding.

/// `build_bbox_lut_1d`: returns `(idx, coef, lut_size, bin_centers)`.
/// `delta_pos0 = None` is the `("no", "lut", …)` no-split case.
#[pyfunction]
#[pyo3(signature = (pos0, *, delta_pos0=None, mask=None, bins, allow_pos0_neg=false))]
fn build_bbox_lut_1d<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    delta_pos0: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: usize,
    allow_pos0_neg: bool,
) -> PyResult<Lut1dPy<'py>> {
    let pos0_s = as_slice_1d(&pos0)?;
    let delta_s = match &delta_pos0 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    let (lut, centers) = rs_build_bbox_lut_1d(
        pos0_s,
        delta_s,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        None,
        None,
    );
    Ok(lut_1d_tuple(py, lut, centers))
}

/// `build_bbox_lut_2d`: returns `(idx, coef, lut_size, bin_centers0, bin_centers1)`.
/// `delta_pos0`/`delta_pos1` both given (bbox split) or both omitted (no-split).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    pos0, pos1, *, delta_pos0=None, delta_pos1=None, mask=None, bins,
    allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period,
))]
fn build_bbox_lut_2d<'py>(
    py: Python<'py>,
    pos0: PyReadonlyArray1<'py, f64>,
    pos1: PyReadonlyArray1<'py, f64>,
    delta_pos0: Option<PyReadonlyArray1<'py, f64>>,
    delta_pos1: Option<PyReadonlyArray1<'py, f64>>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Lut2dPy<'py>> {
    let d0 = match &delta_pos0 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    let d1 = match &delta_pos1 {
        Some(d) => Some(as_slice_1d(d)?),
        None => None,
    };
    if d0.is_some() != d1.is_some() {
        return Err(PyValueError::new_err(
            "delta_pos0 and delta_pos1 must both be given (bbox split) or both omitted (no-split)",
        ));
    }
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let (lut, bc0, bc1) = rs_build_bbox_lut_2d(
        as_slice_1d(&pos0)?,
        d0,
        as_slice_1d(&pos1)?,
        d1,
        mask_slice(&mask)?,
        bins,
        &bounds,
    );
    Ok(lut_2d_tuple(py, lut, bc0, bc1))
}

/// `build_full_lut_1d`: `corners` pre-flattened to f64 (length `8*npix`).
/// Returns `(idx, coef, lut_size, bin_centers)`.
#[pyfunction]
#[pyo3(signature = (corners, *, mask=None, bins, allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period))]
fn build_full_lut_1d<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: usize,
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Lut1dPy<'py>> {
    let (lut, centers) = rs_build_full_lut_1d(
        as_slice_1d(&corners)?,
        mask_slice(&mask)?,
        bins,
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        None,
        None,
    );
    Ok(lut_1d_tuple(py, lut, centers))
}

/// `build_full_lut_2d`: `corners` pre-flattened to f64 (length `8*npix`).
/// Returns `(idx, coef, lut_size, bin_centers0, bin_centers1)`.
#[pyfunction]
#[pyo3(signature = (corners, *, mask=None, bins, allow_pos0_neg=false, chi_disc_at_pi=true, pos1_period))]
fn build_full_lut_2d<'py>(
    py: Python<'py>,
    corners: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, i8>>,
    bins: (usize, usize),
    allow_pos0_neg: bool,
    chi_disc_at_pi: bool,
    pos1_period: f64,
) -> PyResult<Lut2dPy<'py>> {
    let bounds = Bbox2dBounds {
        allow_pos0_neg,
        chi_disc_at_pi,
        pos1_period,
        // Ranges are a high-level orchestration concern; the raw-kernel wrappers
        // expose the full data extent (no override).
        radial_range: None,
        azimuth_range: None,
    };
    let (lut, bc0, bc1) =
        rs_build_full_lut_2d(as_slice_1d(&corners)?, mask_slice(&mask)?, bins, &bounds);
    Ok(lut_2d_tuple(py, lut, bc0, bc1))
}

/// Reassemble a [`Lut`] from numpy `idx`/`coef` (flattened `(n_bins, lut_size)`)
/// and the row width `lut_size`.
fn lut_from_parts(
    idx: &PyReadonlyArray1<'_, i32>,
    coef: &PyReadonlyArray1<'_, f32>,
    lut_size: usize,
) -> PyResult<Lut> {
    Ok(Lut {
        coef: as_slice_1d(coef)?.to_vec(),
        idx: as_slice_1d(idx)?.to_vec(),
        lut_size,
    })
}

/// `lut_integrate1d`: apply a 1D dense LUT to a preprocessed `(npix, 4)` array.
/// Returns a dict of the `Integrate1dtpl` fields.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (idx, coef, lut_size, prep, bin_centers, *, error_model=0, empty=0.0))]
fn lut_integrate1d<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray1<'py, i32>,
    coef: PyReadonlyArray1<'py, f32>,
    lut_size: usize,
    prep: PyReadonlyArray2<'py, f32>,
    bin_centers: PyReadonlyArray1<'py, f64>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let lut = lut_from_parts(&idx, &coef, lut_size)?;
    let prep_s = as_slice_2d(&prep)?;
    let centers = as_slice_1d(&bin_centers)?.to_vec();
    let em = self::error_model(error_model)?;
    let r = rs_lut_integrate1d(&lut, prep_s, centers, em, empty);
    csr_integrate1d_to_dict(py, &r)
}

/// `lut_integrate2d`: apply a 2D dense LUT. Returns a dict of the
/// `Integrate2dtpl` fields (flat (azimuthal, radial) C-order).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (idx, coef, lut_size, prep, bin_centers0, bin_centers1, *, error_model=0, empty=0.0))]
fn lut_integrate2d<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray1<'py, i32>,
    coef: PyReadonlyArray1<'py, f32>,
    lut_size: usize,
    prep: PyReadonlyArray2<'py, f32>,
    bin_centers0: PyReadonlyArray1<'py, f64>,
    bin_centers1: PyReadonlyArray1<'py, f64>,
    error_model: i32,
    empty: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let lut = lut_from_parts(&idx, &coef, lut_size)?;
    let prep_s = as_slice_2d(&prep)?;
    let bc0 = as_slice_1d(&bin_centers0)?.to_vec();
    let bc1 = as_slice_1d(&bin_centers1)?.to_vec();
    let em = self::error_model(error_model)?;
    let r = rs_lut_integrate2d(&lut, prep_s, bc0, bc1, em, empty);
    integrate2d_to_dict(py, &r)
}

// --------------------------------------------------------------------------
// High-level integrator (drop-in)
// --------------------------------------------------------------------------

/// A pure-Rust drop-in for `pyFAI.integrator.AzimuthalIntegrator`, exposing the
/// no-split histogram `integrate1d`/`integrate2d` path. Construct via
/// `AzimuthalIntegrator.load(poni)`; the detector model is resolved from the
/// PONI file's `Detector:` name (currently Pilatus1M).
///
/// Unlike the per-kernel functions, this regenerates pixel positions,
/// corrections, gap mask, dummy, and preproc rows itself from the geometry and
/// the image — the same chain `rsfai_engine::AzimuthalIntegrator` runs — so a
/// parity test feeds it only the PONI and the frame.
#[pyclass(name = "AzimuthalIntegrator")]
struct PyAzimuthalIntegrator {
    inner: RsAzimuthalIntegrator,
}

#[pymethods]
impl PyAzimuthalIntegrator {
    /// Load from a `.poni` file, resolving the detector from its `Detector:`
    /// name. Raises `ValueError` if the file cannot be parsed or the detector is
    /// not one with a golden-validated path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = RsAzimuthalIntegrator::load(path)
            .map_err(|e| PyValueError::new_err(format!("failed to load {path:?}: {e}")))?;
        Ok(Self { inner })
    }

    /// 1D integration of a detector `image` (an `(slow, fast)` f32 frame) into
    /// `npt` radial bins in `unit`, using the `method` `(split, algo[, impl])`
    /// tuple (default `("no", "histogram")`). Returns a dict keyed like
    /// `pyFAI.containers.Integrate1dResult` (`radial`, `intensity`, `sigma`,
    /// `count`, `sum_signal`, `sum_variance`, `sum_normalization`,
    /// `sum_normalization2`, `std`, `sem`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        image, npt, unit, *, method=None, correct_solid_angle=true,
        polarization_factor=None, normalization_factor=1.0, error_model=0,
        radial_range=None, azimuth_range=None,
    ))]
    fn integrate1d<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        npt: usize,
        unit: &str,
        method: Option<Vec<String>>,
        correct_solid_angle: bool,
        polarization_factor: Option<f64>,
        normalization_factor: f32,
        error_model: i32,
        radial_range: Option<(f64, f64)>,
        azimuth_range: Option<(f64, f64)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let data = self.image_slice(&image)?;
        let m = parse_method(method.as_deref())?;
        let opts = self.options(
            correct_solid_angle,
            polarization_factor,
            normalization_factor,
            error_model,
            m,
            radial_range,
            azimuth_range,
        )?;
        let r = self.inner.integrate1d(data, npt, radial_unit(unit)?, &opts);
        // No-split histogram is the only 1D engine whose pyFAI accumulators are
        // f32; the sparse and split-histogram engines emit f64.
        let acc_f32 = m.split == Split::No && m.algo == Algo::Histogram;
        integrate1d_result_to_dict(py, &r, acc_f32)
    }

    /// 2D integration of a detector `image` into a `(npt_azim, npt_rad)` cake,
    /// radial in `unit`, azimuth in degrees, using the `method` split + algo.
    /// Returns a dict keyed like `pyFAI.containers.Integrate2dResult`; the
    /// per-cell arrays are 2D, shaped `(npt_azim, npt_rad)` to match pyFAI.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        image, npt_rad, npt_azim, unit, *, method=None, correct_solid_angle=true,
        polarization_factor=None, normalization_factor=1.0, error_model=0,
        radial_range=None, azimuth_range=None,
    ))]
    fn integrate2d<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        npt_rad: usize,
        npt_azim: usize,
        unit: &str,
        method: Option<Vec<String>>,
        correct_solid_angle: bool,
        polarization_factor: Option<f64>,
        normalization_factor: f32,
        error_model: i32,
        radial_range: Option<(f64, f64)>,
        azimuth_range: Option<(f64, f64)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let data = self.image_slice(&image)?;
        let m = parse_method(method.as_deref())?;
        let opts = self.options(
            correct_solid_angle,
            polarization_factor,
            normalization_factor,
            error_model,
            m,
            radial_range,
            azimuth_range,
        )?;
        let r = self
            .inner
            .integrate2d(data, npt_rad, npt_azim, radial_unit(unit)?, &opts);
        // Every 2D engine emits f64 accumulators (matching pyFAI), so the 2D
        // dict needs no per-method dtype gate.
        integrate2d_result_to_dict(py, &r)
    }
}

impl PyAzimuthalIntegrator {
    /// Borrow the image as its flat C-order slice, rejecting a non-contiguous or
    /// wrong-sized frame at the FFI boundary (so a shape error surfaces as a
    /// `ValueError`, not an engine panic).
    fn image_slice<'a>(&self, image: &'a PyReadonlyArray2<'_, f32>) -> PyResult<&'a [f32]> {
        let data = image
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("image must be C-contiguous: {e}")))?;
        let expected = self.inner.detector.size();
        if data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "image has {} pixels but detector expects {expected}",
                data.len()
            )));
        }
        Ok(data)
    }

    /// Assemble [`IntegrationOptions`] from the keyword arguments.
    #[allow(clippy::too_many_arguments)]
    fn options(
        &self,
        correct_solid_angle: bool,
        polarization_factor: Option<f64>,
        normalization_factor: f32,
        error_model_code: i32,
        method: Method,
        radial_range: Option<(f64, f64)>,
        azimuth_range: Option<(f64, f64)>,
    ) -> PyResult<IntegrationOptions> {
        Ok(IntegrationOptions {
            correct_solid_angle,
            polarization_factor,
            normalization_factor,
            error_model: error_model(error_model_code)?,
            method,
            radial_range,
            azimuth_range,
        })
    }
}

// --------------------------------------------------------------------------
// Output conversion helpers
// --------------------------------------------------------------------------

fn csr_1d_tuple<'py>(py: Python<'py>, csr: Csr, centers: Vec<f64>) -> Csr1dPy<'py> {
    (
        csr.data.into_pyarray(py),
        csr.indices.into_pyarray(py),
        csr.indptr.into_pyarray(py),
        centers.into_pyarray(py),
    )
}

fn csr_2d_tuple<'py>(py: Python<'py>, csr: Csr, bc0: Vec<f64>, bc1: Vec<f64>) -> Csr2dPy<'py> {
    (
        csr.data.into_pyarray(py),
        csr.indices.into_pyarray(py),
        csr.indptr.into_pyarray(py),
        bc0.into_pyarray(py),
        bc1.into_pyarray(py),
    )
}

fn csc_1d_tuple<'py>(py: Python<'py>, csc: Csc, centers: Vec<f64>) -> Csr1dPy<'py> {
    (
        csc.data.into_pyarray(py),
        csc.indices.into_pyarray(py),
        csc.indptr.into_pyarray(py),
        centers.into_pyarray(py),
    )
}

fn csc_2d_tuple<'py>(py: Python<'py>, csc: Csc, bc0: Vec<f64>, bc1: Vec<f64>) -> Csr2dPy<'py> {
    (
        csc.data.into_pyarray(py),
        csc.indices.into_pyarray(py),
        csc.indptr.into_pyarray(py),
        bc0.into_pyarray(py),
        bc1.into_pyarray(py),
    )
}

fn lut_1d_tuple<'py>(py: Python<'py>, lut: Lut, centers: Vec<f64>) -> Lut1dPy<'py> {
    (
        lut.idx.into_pyarray(py),
        lut.coef.into_pyarray(py),
        lut.lut_size,
        centers.into_pyarray(py),
    )
}

fn lut_2d_tuple<'py>(py: Python<'py>, lut: Lut, bc0: Vec<f64>, bc1: Vec<f64>) -> Lut2dPy<'py> {
    (
        lut.idx.into_pyarray(py),
        lut.coef.into_pyarray(py),
        lut.lut_size,
        bc0.into_pyarray(py),
        bc1.into_pyarray(py),
    )
}

fn integrate1d_to_dict<'py>(py: Python<'py>, r: &Integrate1d) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("position", r.position.clone().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.clone().into_pyarray(py))?;
    d.set_item("sigma", r.sigma.clone().into_pyarray(py))?;
    d.set_item("signal", r.signal.clone().into_pyarray(py))?;
    d.set_item("variance", r.variance.clone().into_pyarray(py))?;
    d.set_item("normalization", r.normalization.clone().into_pyarray(py))?;
    d.set_item("count", r.count.clone().into_pyarray(py))?;
    d.set_item("std", r.std.clone().into_pyarray(py))?;
    d.set_item("sem", r.sem.clone().into_pyarray(py))?;
    d.set_item("norm_sq", r.norm_sq.clone().into_pyarray(py))?;
    Ok(d)
}

fn csr_integrate1d_to_dict<'py>(
    py: Python<'py>,
    r: &CsrIntegrate1d,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("position", r.position.clone().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.clone().into_pyarray(py))?;
    d.set_item("sigma", r.sigma.clone().into_pyarray(py))?;
    d.set_item("sum_signal", r.sum_signal.clone().into_pyarray(py))?;
    d.set_item("sum_variance", r.sum_variance.clone().into_pyarray(py))?;
    d.set_item(
        "sum_normalization",
        r.sum_normalization.clone().into_pyarray(py),
    )?;
    d.set_item("count", r.count.clone().into_pyarray(py))?;
    d.set_item("std", r.std.clone().into_pyarray(py))?;
    d.set_item("sem", r.sem.clone().into_pyarray(py))?;
    d.set_item("sum_norm_sq", r.sum_norm_sq.clone().into_pyarray(py))?;
    Ok(d)
}

fn integrate2d_to_dict<'py>(py: Python<'py>, r: &Integrate2d) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("radial", r.radial.clone().into_pyarray(py))?;
    d.set_item("azimuthal", r.azimuthal.clone().into_pyarray(py))?;
    d.set_item("bins", r.bins)?;
    d.set_item("intensity", r.intensity.clone().into_pyarray(py))?;
    d.set_item("sigma", r.sigma.clone().into_pyarray(py))?;
    d.set_item("signal", r.signal.clone().into_pyarray(py))?;
    d.set_item("variance", r.variance.clone().into_pyarray(py))?;
    d.set_item("normalization", r.normalization.clone().into_pyarray(py))?;
    d.set_item("count", r.count.clone().into_pyarray(py))?;
    d.set_item("std", r.std.clone().into_pyarray(py))?;
    d.set_item("sem", r.sem.clone().into_pyarray(py))?;
    d.set_item("norm_sq", r.norm_sq.clone().into_pyarray(py))?;
    Ok(d)
}

/// Emit an f64 accumulator vector as the numpy dtype pyFAI's engine produces:
/// **f32** for the no-split histogram (`acc_f32 = true`), **f64** otherwise.
/// `Integrate1dResult` carries the accumulators as f64 (the sparse engines'
/// native width); pyFAI's no-split histogram `Integrate1dtpl` stores `count`/
/// `sum_*` as f32, and the parity harness rejects on a dtype mismatch, so that
/// one path widens to f32 (losslessly — the f32 truncation already happened in
/// the engine) to match.
fn acc_array<'py>(py: Python<'py>, v: &[f64], acc_f32: bool) -> Bound<'py, PyAny> {
    if acc_f32 {
        v.iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>()
            .into_pyarray(py)
            .into_any()
    } else {
        v.to_vec().into_pyarray(py).into_any()
    }
}

/// Build the pyFAI-keyed dict from a high-level 1D result. Keys mirror
/// `pyFAI.containers.Integrate1dResult` attributes so a parity test compares
/// field-by-field; every array is 1D of length `npt`. `acc_f32` selects the
/// accumulator dtype per engine (see [`acc_array`]).
fn integrate1d_result_to_dict<'py>(
    py: Python<'py>,
    r: &Integrate1dResult,
    acc_f32: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("radial", r.radial.clone().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.clone().into_pyarray(py))?;
    d.set_item("sigma", r.sigma.clone().into_pyarray(py))?;
    d.set_item("count", acc_array(py, &r.count, acc_f32))?;
    d.set_item("sum_signal", acc_array(py, &r.sum_signal, acc_f32))?;
    d.set_item("sum_variance", acc_array(py, &r.sum_variance, acc_f32))?;
    d.set_item(
        "sum_normalization",
        acc_array(py, &r.sum_normalization, acc_f32),
    )?;
    d.set_item(
        "sum_normalization2",
        acc_array(py, &r.sum_normalization2, acc_f32),
    )?;
    d.set_item("std", r.std.clone().into_pyarray(py))?;
    d.set_item("sem", r.sem.clone().into_pyarray(py))?;
    Ok(d)
}

/// Reshape a flat `(azimuthal, radial)` C-order vector (the engine's 2D layout)
/// into an `(npt_azim, npt_rad)` numpy array — pyFAI's `Integrate2dResult` cell
/// shape. `bins` is `(radial, azimuthal)`, so the array is `bins.1 × bins.0`.
fn reshape_azim_rad<'py, T: numpy::Element>(
    py: Python<'py>,
    v: Vec<T>,
    bins: (usize, usize),
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let arr = Array2::from_shape_vec((bins.1, bins.0), v)
        .map_err(|e| PyValueError::new_err(format!("2D result reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Build the pyFAI-keyed dict from a high-level 2D result. The bin-center axes
/// (`radial`, `azimuthal`) are 1D; the per-cell fields are 2D `(npt_azim,
/// npt_rad)`, matching `pyFAI.containers.Integrate2dResult`.
fn integrate2d_result_to_dict<'py>(
    py: Python<'py>,
    r: &Integrate2dResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("radial", r.radial.clone().into_pyarray(py))?;
    d.set_item("azimuthal", r.azimuthal.clone().into_pyarray(py))?;
    d.set_item("bins", r.bins)?;
    d.set_item(
        "intensity",
        reshape_azim_rad(py, r.intensity.clone(), r.bins)?,
    )?;
    d.set_item("sigma", reshape_azim_rad(py, r.sigma.clone(), r.bins)?)?;
    d.set_item("count", reshape_azim_rad(py, r.count.clone(), r.bins)?)?;
    d.set_item(
        "sum_signal",
        reshape_azim_rad(py, r.sum_signal.clone(), r.bins)?,
    )?;
    d.set_item(
        "sum_variance",
        reshape_azim_rad(py, r.sum_variance.clone(), r.bins)?,
    )?;
    d.set_item(
        "sum_normalization",
        reshape_azim_rad(py, r.sum_normalization.clone(), r.bins)?,
    )?;
    d.set_item(
        "sum_normalization2",
        reshape_azim_rad(py, r.sum_normalization2.clone(), r.bins)?,
    )?;
    d.set_item("std", reshape_azim_rad(py, r.std.clone(), r.bins)?)?;
    d.set_item("sem", reshape_azim_rad(py, r.sem.clone(), r.bins)?)?;
    Ok(d)
}

// --------------------------------------------------------------------------
// Module
// --------------------------------------------------------------------------

#[pymodule]
fn rsfai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Bit-exact rsFAI integration kernels (PyO3 bindings)",
    )?;
    m.add_function(wrap_pyfunction!(preproc4, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_preproc, m)?)?;
    m.add_function(wrap_pyfunction!(histogram1d, m)?)?;
    m.add_function(wrap_pyfunction!(histogram2d, m)?)?;
    m.add_function(wrap_pyfunction!(histogram1d_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(histogram2d_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(histogram1d_full, m)?)?;
    m.add_function(wrap_pyfunction!(histogram2d_full, m)?)?;
    m.add_function(wrap_pyfunction!(histogram2d_pseudo, m)?)?;
    m.add_function(wrap_pyfunction!(build_bbox_csr_1d, m)?)?;
    m.add_function(wrap_pyfunction!(build_bbox_csr_2d, m)?)?;
    m.add_function(wrap_pyfunction!(build_full_csr_1d, m)?)?;
    m.add_function(wrap_pyfunction!(build_full_csr_2d, m)?)?;
    m.add_function(wrap_pyfunction!(csr_integrate1d, m)?)?;
    m.add_function(wrap_pyfunction!(csr_integrate2d, m)?)?;
    m.add_function(wrap_pyfunction!(build_bbox_csc_1d, m)?)?;
    m.add_function(wrap_pyfunction!(build_bbox_csc_2d, m)?)?;
    m.add_function(wrap_pyfunction!(build_full_csc_1d, m)?)?;
    m.add_function(wrap_pyfunction!(build_full_csc_2d, m)?)?;
    m.add_function(wrap_pyfunction!(csc_integrate1d, m)?)?;
    m.add_function(wrap_pyfunction!(csc_integrate2d, m)?)?;
    m.add_function(wrap_pyfunction!(build_bbox_lut_1d, m)?)?;
    m.add_function(wrap_pyfunction!(build_bbox_lut_2d, m)?)?;
    m.add_function(wrap_pyfunction!(build_full_lut_1d, m)?)?;
    m.add_function(wrap_pyfunction!(build_full_lut_2d, m)?)?;
    m.add_function(wrap_pyfunction!(lut_integrate1d, m)?)?;
    m.add_function(wrap_pyfunction!(lut_integrate2d, m)?)?;
    m.add_class::<PyAzimuthalIntegrator>()?;
    Ok(())
}
