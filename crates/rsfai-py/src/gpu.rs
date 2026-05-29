//! GPU (OpenCL) bindings: a cached [`GpuEngine`] mirroring the CPU `Engine`,
//! plus device discovery.
//!
//! Unlike the CPU path (where the caller runs `preproc4` and the engine consumes
//! a preprocessed `(npix, 4)` array), pyFAI's OpenCL kernels run the per-pixel
//! corrections **on the device**. So the GPU engine takes the **raw image** each
//! frame and the corrections (solid angle, polarization, flat, dark, mask, …) are
//! detector/geometry-fixed: they are supplied ONCE at construction and uploaded
//! with the matrix, exactly the "build once, loop frames" streaming shape the
//! cached engine is for. `integrate1d`/`integrate2d` then take only the per-frame
//! image and return a result dict shaped like the CPU engine's.
//!
//! All three OpenCL algos are exposed: CSR and LUT (deterministic on the M4 Pro,
//! validated bit-exact) and histogram (atomic scatter — `count` is bit-exact via
//! integer atomics, the float columns carry a bounded atomic-add noise; pyFAI is
//! not bit-reproducible against itself there either). See `rsfai-opencl`.
//!
//! Results are **tolerance-validated, not bit-exact** in general: the target
//! device (Apple M4 Pro) has no f64, so pyFAI's kernels accumulate in a doubleword
//! (two-f32) representation and the work-group reductions reorder additions.

use numpy::{IntoPyArray, PyReadonlyArray1};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::program::Program;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rsfai_opencl::program::{build_csr_program, build_histogram_program, build_lut_program};
use rsfai_opencl::{
    create_queue, default_context, integrate1d_csr, integrate1d_histogram, integrate1d_lut,
    integrate2d_csr, integrate2d_histogram, integrate2d_lut, ClSession, Corrections4Args,
    Corrections4aArgs, CsrInputs, HistResult1d, HistResult2d, HistogramInputs, HistogramScalars,
    LutInputs, Result1d, Result2d,
};

use crate::as_slice_1d;

/// Map an `rsfai-opencl` `String` error to a Python `RuntimeError`.
fn ocl_err<T>(r: Result<T, String>, what: &str) -> PyResult<T> {
    r.map_err(|e| PyRuntimeError::new_err(format!("{what}: {e}")))
}

/// Copy an optional readonly f32 array to an owned `Vec` (`None` stays `None`).
fn opt_vec_f32(a: Option<PyReadonlyArray1<'_, f32>>) -> PyResult<Option<Vec<f32>>> {
    match a {
        Some(arr) => Ok(Some(as_slice_1d(&arr)?.to_vec())),
        None => Ok(None),
    }
}

/// Copy an optional readonly i8 array (the mask) to an owned `Vec`.
fn opt_vec_i8(a: Option<PyReadonlyArray1<'_, i8>>) -> PyResult<Option<Vec<i8>>> {
    match a {
        Some(arr) => Ok(Some(as_slice_1d(&arr)?.to_vec())),
        None => Ok(None),
    }
}

/// Enumerate every OpenCL device visible on this machine, one dict per device
/// (`name`, `vendor`, `type`, `is_gpu`, `double_fp_config`,
/// `max_work_group_size`). `double_fp_config == 0` means the device has no f64
/// and pyFAI's doubleword kernels are required (the Apple M4 Pro case).
#[pyfunction]
pub fn list_gpu_devices(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    let devs = rsfai_opencl::list_devices()
        .map_err(|e| PyRuntimeError::new_err(format!("OpenCL device enumeration failed: {e}")))?;
    let mut out = Vec::with_capacity(devs.len());
    for d in devs {
        let dict = PyDict::new(py);
        dict.set_item("name", d.name)?;
        dict.set_item("vendor", d.vendor)?;
        dict.set_item("type", d.dev_type)?;
        dict.set_item("is_gpu", d.dev_type == 4)?;
        dict.set_item("double_fp_config", d.double_fp_config)?;
        dict.set_item("max_work_group_size", d.max_work_group_size)?;
        out.push(dict.unbind());
    }
    Ok(out)
}

/// The detector/geometry-fixed correction config, captured at construction. The
/// `do_*` flags are derived from which correction arrays were supplied; the
/// scalars come from the constructor kwargs. `dtype` is only consumed by the CSR
/// kernel (`corrections4a`); the LUT/histogram kernels (`corrections4`) pre-cast
/// the image to float and ignore it.
#[derive(Clone, Copy)]
struct CorrConfig {
    error_model: i8,
    dtype: i8,
    do_dark: i8,
    do_dark_variance: i8,
    do_flat: i8,
    do_solidangle: i8,
    do_polarization: i8,
    do_absorption: i8,
    do_mask: i8,
    do_dummy: i8,
    dummy: f32,
    delta_dummy: f32,
    normalization_factor: f32,
    apply_normalization: i8,
}

impl CorrConfig {
    /// The CSR `corrections4a` argument bundle (carries `dtype`).
    fn to_csr_args(self) -> Corrections4aArgs {
        Corrections4aArgs {
            dtype: self.dtype,
            error_model: self.error_model,
            do_dark: self.do_dark,
            do_dark_variance: self.do_dark_variance,
            do_flat: self.do_flat,
            do_solidangle: self.do_solidangle,
            do_polarization: self.do_polarization,
            do_absorption: self.do_absorption,
            do_mask: self.do_mask,
            do_dummy: self.do_dummy,
            dummy: self.dummy,
            delta_dummy: self.delta_dummy,
            normalization_factor: self.normalization_factor,
            apply_normalization: self.apply_normalization,
        }
    }

    /// The LUT/histogram `corrections4` argument bundle (no `dtype`).
    fn to_corr4_args(self) -> Corrections4Args {
        Corrections4Args {
            error_model: self.error_model,
            do_dark: self.do_dark,
            do_dark_variance: self.do_dark_variance,
            do_flat: self.do_flat,
            do_solidangle: self.do_solidangle,
            do_polarization: self.do_polarization,
            do_absorption: self.do_absorption,
            do_mask: self.do_mask,
            do_dummy: self.do_dummy,
            dummy: self.dummy,
            delta_dummy: self.delta_dummy,
            normalization_factor: self.normalization_factor,
            apply_normalization: self.apply_normalization,
        }
    }
}

/// The cached sparse/scatter matrix on the GPU side, one variant per algo. Holds
/// the host-side data the `rsfai-opencl` orchestrator uploads on each frame plus
/// the algo's work-group parameter.
enum GpuMatrix {
    Csr {
        data: Vec<f32>,
        indices: Vec<i32>,
        indptr: Vec<i32>,
        wg_min: usize,
    },
    Lut {
        idx: Vec<i32>,
        coef: Vec<f32>,
        lut_size: usize,
        block_size: usize,
    },
    Histogram {
        radial: Vec<f32>,
        azimuthal: Vec<f32>,
        radial_mini: f32,
        radial_maxi: f32,
        azim_mini: f32,
        azim_maxi: f32,
        check_azim: i8,
        block_size: usize,
    },
}

/// The detector/geometry-fixed corrections, owned and re-uploaded each frame by
/// the kernel orchestrator.
#[derive(Default)]
struct CorrArrays {
    variance: Option<Vec<f32>>,
    dark: Option<Vec<f32>>,
    dark_variance: Option<Vec<f32>>,
    flat: Option<Vec<f32>>,
    solidangle: Option<Vec<f32>>,
    polarization: Option<Vec<f32>>,
    absorption: Option<Vec<f32>>,
    mask: Option<Vec<i8>>,
}

/// A cached OpenCL integration engine: the compiled program + context + queue +
/// matrix + fixed corrections, all owned Rust-side and reused across frames.
/// Mirror of the CPU `Engine`; 1D vs 2D is set by whether a second bin-center
/// axis was supplied at construction. Marked `unsendable` because the OpenCL
/// handles are not `Send`/`Sync` (raw device pointers) — use one engine per
/// thread.
#[pyclass(unsendable)]
pub struct GpuEngine {
    // OpenCL handles, owned together (program built against context, queue bound
    // to it); ClSession borrows all three per call.
    context: Context,
    queue: CommandQueue,
    program: Program,
    matrix: GpuMatrix,
    corr: CorrConfig,
    arrays: CorrArrays,
    empty: f32,
    image_size: usize,
    centers0: Vec<f64>,
    centers1: Option<Vec<f64>>,
}

impl GpuEngine {
    /// Total bin count from the bin-center axes (2D ⇒ product of both).
    fn bins(centers0: &[f64], centers1: &Option<Vec<f64>>) -> usize {
        match centers1 {
            Some(c1) => centers0.len() * c1.len(),
            None => centers0.len(),
        }
    }

    /// Build a CSR engine: create the (GPU-preferred) context + queue, compile the
    /// CSR program for `bins`/`image_size`, and capture the matrix + corrections.
    #[allow(clippy::too_many_arguments)]
    fn build_csr(
        data: Vec<f32>,
        indices: Vec<i32>,
        indptr: Vec<i32>,
        centers0: Vec<f64>,
        centers1: Option<Vec<f64>>,
        image_size: usize,
        corr: CorrConfig,
        arrays: CorrArrays,
        empty: f32,
        wg_min: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let bins = Self::bins(&centers0, &centers1);
        if indptr.len() != bins + 1 {
            return Err(PyValueError::new_err(format!(
                "csr indptr length {} must be bins + 1 = {}",
                indptr.len(),
                bins + 1
            )));
        }
        if indices.len() != data.len() {
            return Err(PyValueError::new_err(
                "csr data and indices must have the same length",
            ));
        }
        let (context, queue) = open_device(prefer_gpu)?;
        let program = ocl_err(
            build_csr_program(&context, bins, image_size),
            "build CSR program",
        )?;
        Ok(Self {
            context,
            queue,
            program,
            matrix: GpuMatrix::Csr {
                data,
                indices,
                indptr,
                wg_min,
            },
            corr,
            arrays,
            empty,
            image_size,
            centers0,
            centers1,
        })
    }

    /// Build a LUT engine: compile the LUT program for `bins`/`image_size`/
    /// `lut_size` and capture the densified matrix + corrections.
    #[allow(clippy::too_many_arguments)]
    fn build_lut(
        idx: Vec<i32>,
        coef: Vec<f32>,
        lut_size: usize,
        centers0: Vec<f64>,
        centers1: Option<Vec<f64>>,
        image_size: usize,
        corr: CorrConfig,
        arrays: CorrArrays,
        empty: f32,
        block_size: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let bins = Self::bins(&centers0, &centers1);
        if idx.len() != bins * lut_size || coef.len() != bins * lut_size {
            return Err(PyValueError::new_err(format!(
                "lut idx/coef length must be bins * lut_size = {} * {} = {}",
                bins,
                lut_size,
                bins * lut_size
            )));
        }
        let (context, queue) = open_device(prefer_gpu)?;
        let program = ocl_err(
            build_lut_program(&context, bins, image_size, lut_size),
            "build LUT program",
        )?;
        Ok(Self {
            context,
            queue,
            program,
            matrix: GpuMatrix::Lut {
                idx,
                coef,
                lut_size,
                block_size,
            },
            corr,
            arrays,
            empty,
            image_size,
            centers0,
            centers1,
        })
    }

    /// Build a histogram engine: compile the histogram program for `bins`/
    /// `image_size`/`block_size` and capture the per-pixel position arrays, the
    /// range scalars, and the corrections.
    #[allow(clippy::too_many_arguments)]
    fn build_histogram(
        radial: Vec<f32>,
        azimuthal: Vec<f32>,
        radial_mini: f32,
        radial_maxi: f32,
        azim_mini: f32,
        azim_maxi: f32,
        check_azim: i8,
        centers0: Vec<f64>,
        centers1: Option<Vec<f64>>,
        image_size: usize,
        corr: CorrConfig,
        arrays: CorrArrays,
        empty: f32,
        block_size: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let bins = Self::bins(&centers0, &centers1);
        if radial.len() != image_size || azimuthal.len() != image_size {
            return Err(PyValueError::new_err(format!(
                "radial/azimuthal length must equal image_size = {image_size}"
            )));
        }
        let (context, queue) = open_device(prefer_gpu)?;
        let program = ocl_err(
            build_histogram_program(&context, bins, image_size, block_size),
            "build histogram program",
        )?;
        Ok(Self {
            context,
            queue,
            program,
            matrix: GpuMatrix::Histogram {
                radial,
                azimuthal,
                radial_mini,
                radial_maxi,
                azim_mini,
                azim_maxi,
                check_azim,
                block_size,
            },
            corr,
            arrays,
            empty,
            image_size,
            centers0,
            centers1,
        })
    }

    /// Borrow the OpenCL handles as a [`ClSession`] for one integration call.
    fn session(&self) -> ClSession<'_> {
        ClSession {
            context: &self.context,
            queue: &self.queue,
            program: &self.program,
        }
    }

    /// Validate `image` matches the engine and return it as a slice.
    fn check_image<'a>(&self, image: &'a PyReadonlyArray1<'_, i32>) -> PyResult<&'a [i32]> {
        let img = as_slice_1d(image)?;
        if img.len() != self.image_size {
            return Err(PyValueError::new_err(format!(
                "image length {} != engine image_size {}",
                img.len(),
                self.image_size
            )));
        }
        Ok(img)
    }

    /// Assemble per-frame [`CsrInputs`] from the image plus the engine's fixed
    /// correction arrays and CSR matrix.
    fn csr_inputs<'a>(
        &'a self,
        image: &'a [i32],
        data: &'a [f32],
        indices: &'a [i32],
        indptr: &'a [i32],
    ) -> CsrInputs<'a> {
        CsrInputs {
            image_i32: image,
            variance: self.arrays.variance.as_deref(),
            dark: self.arrays.dark.as_deref(),
            dark_variance: self.arrays.dark_variance.as_deref(),
            flat: self.arrays.flat.as_deref(),
            solidangle: self.arrays.solidangle.as_deref(),
            polarization: self.arrays.polarization.as_deref(),
            absorption: self.arrays.absorption.as_deref(),
            mask: self.arrays.mask.as_deref(),
            csr_data: data,
            csr_indices: indices,
            csr_indptr: indptr,
        }
    }

    /// Assemble per-frame [`LutInputs`] from the image plus the engine's fixed
    /// correction arrays and densified LUT.
    fn lut_inputs<'a>(
        &'a self,
        image: &'a [i32],
        idx: &'a [i32],
        coef: &'a [f32],
        lut_size: usize,
        bins: usize,
    ) -> LutInputs<'a> {
        LutInputs {
            image_i32: image,
            variance: self.arrays.variance.as_deref(),
            dark: self.arrays.dark.as_deref(),
            dark_variance: self.arrays.dark_variance.as_deref(),
            flat: self.arrays.flat.as_deref(),
            solidangle: self.arrays.solidangle.as_deref(),
            polarization: self.arrays.polarization.as_deref(),
            absorption: self.arrays.absorption.as_deref(),
            mask: self.arrays.mask.as_deref(),
            lut_idx: idx,
            lut_coef: coef,
            bins,
            lut_size,
        }
    }

    /// Assemble per-frame [`HistogramInputs`] from the image plus the engine's
    /// fixed correction arrays and per-pixel position arrays.
    fn hist_inputs<'a>(
        &'a self,
        image: &'a [i32],
        radial: &'a [f32],
        azimuthal: &'a [f32],
        bins: usize,
    ) -> HistogramInputs<'a> {
        HistogramInputs {
            image_i32: image,
            variance: self.arrays.variance.as_deref(),
            dark: self.arrays.dark.as_deref(),
            dark_variance: self.arrays.dark_variance.as_deref(),
            flat: self.arrays.flat.as_deref(),
            solidangle: self.arrays.solidangle.as_deref(),
            polarization: self.arrays.polarization.as_deref(),
            absorption: self.arrays.absorption.as_deref(),
            mask: self.arrays.mask.as_deref(),
            radial,
            azimuthal,
            bins,
        }
    }
}

/// Create a (GPU-preferred) context + command queue, mapping OpenCL errors to
/// Python exceptions.
fn open_device(prefer_gpu: bool) -> PyResult<(Context, CommandQueue)> {
    let (context, _device) = ocl_err(
        default_context(prefer_gpu).map_err(|e| e.to_string()),
        "create OpenCL context",
    )?;
    let queue = ocl_err(
        create_queue(&context).map_err(|e| e.to_string()),
        "create command queue",
    )?;
    Ok((context, queue))
}

/// Build the [`CorrArrays`] (owned copies) + [`CorrConfig`] (scalars + `do_*`
/// flags derived from array presence) shared by every constructor.
#[allow(clippy::too_many_arguments)]
fn corr_from_kwargs(
    variance: Option<PyReadonlyArray1<'_, f32>>,
    dark: Option<PyReadonlyArray1<'_, f32>>,
    dark_variance: Option<PyReadonlyArray1<'_, f32>>,
    flat: Option<PyReadonlyArray1<'_, f32>>,
    solidangle: Option<PyReadonlyArray1<'_, f32>>,
    polarization: Option<PyReadonlyArray1<'_, f32>>,
    absorption: Option<PyReadonlyArray1<'_, f32>>,
    mask: Option<PyReadonlyArray1<'_, i8>>,
    error_model: i32,
    dtype: i32,
    dummy: Option<f32>,
    delta_dummy: f32,
    normalization_factor: f32,
    apply_normalization: bool,
) -> PyResult<(CorrArrays, CorrConfig)> {
    let arrays = CorrArrays {
        variance: opt_vec_f32(variance)?,
        dark: opt_vec_f32(dark)?,
        dark_variance: opt_vec_f32(dark_variance)?,
        flat: opt_vec_f32(flat)?,
        solidangle: opt_vec_f32(solidangle)?,
        polarization: opt_vec_f32(polarization)?,
        absorption: opt_vec_f32(absorption)?,
        mask: opt_vec_i8(mask)?,
    };
    let flag = |b: bool| if b { 1i8 } else { 0i8 };
    let corr = CorrConfig {
        error_model: error_model as i8,
        dtype: dtype as i8,
        do_dark: flag(arrays.dark.is_some()),
        do_dark_variance: flag(arrays.dark_variance.is_some()),
        do_flat: flag(arrays.flat.is_some()),
        do_solidangle: flag(arrays.solidangle.is_some()),
        do_polarization: flag(arrays.polarization.is_some()),
        do_absorption: flag(arrays.absorption.is_some()),
        do_mask: flag(arrays.mask.is_some()),
        do_dummy: flag(dummy.is_some()),
        dummy: dummy.unwrap_or(0.0),
        delta_dummy,
        normalization_factor,
        apply_normalization: flag(apply_normalization),
    };
    Ok((arrays, corr))
}

/// Pack a [`Result1d`] (CSR/LUT) into the CPU-isomorphic `Integrate1dtpl` dict.
fn result1d_dict<'py>(
    py: Python<'py>,
    centers0: &[f64],
    r: Result1d,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("position", centers0.to_vec().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.into_pyarray(py))?;
    d.set_item("sigma", r.sem.clone().into_pyarray(py))?;
    d.set_item("sum_signal", r.signal.into_pyarray(py))?;
    d.set_item("sum_variance", r.variance.into_pyarray(py))?;
    d.set_item("sum_normalization", r.normalization.into_pyarray(py))?;
    d.set_item("count", r.count.into_pyarray(py))?;
    d.set_item("std", r.std.into_pyarray(py))?;
    d.set_item("sem", r.sem.into_pyarray(py))?;
    d.set_item("sum_norm_sq", r.norm_sq.into_pyarray(py))?;
    Ok(d)
}

/// Pack a [`Result2d`] (CSR/LUT) into the CPU-isomorphic `Integrate2dtpl` dict.
fn result2d_dict<'py>(
    py: Python<'py>,
    centers0: &[f64],
    centers1: &[f64],
    r: Result2d,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bins", (centers0.len(), centers1.len()))?;
    d.set_item("radial", centers0.to_vec().into_pyarray(py))?;
    d.set_item("azimuthal", centers1.to_vec().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.into_pyarray(py))?;
    d.set_item("sigma", r.sem.clone().into_pyarray(py))?;
    d.set_item("signal", r.signal.into_pyarray(py))?;
    d.set_item("variance", r.variance.into_pyarray(py))?;
    d.set_item("normalization", r.normalization.into_pyarray(py))?;
    d.set_item("count", r.count.into_pyarray(py))?;
    d.set_item("std", r.std.into_pyarray(py))?;
    d.set_item("sem", r.sem.into_pyarray(py))?;
    d.set_item("norm_sq", r.norm_sq.into_pyarray(py))?;
    Ok(d)
}

/// Pack a [`HistResult1d`] into pyFAI's 1D histogram field dict. The 1D histogram
/// path exposes neither `std`/`sem` nor `norm_sq` (pyFAI does not either); `count`
/// is `uint32` (integer atomics).
fn histresult1d_dict<'py>(
    py: Python<'py>,
    centers0: &[f64],
    r: HistResult1d,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("position", centers0.to_vec().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.into_pyarray(py))?;
    d.set_item("sigma", r.sigma.into_pyarray(py))?;
    d.set_item("count", r.count.into_pyarray(py))?;
    d.set_item("sum_signal", r.signal.into_pyarray(py))?;
    d.set_item("sum_variance", r.variance.into_pyarray(py))?;
    d.set_item("sum_normalization", r.normalization.into_pyarray(py))?;
    Ok(d)
}

/// Pack a [`HistResult2d`] into pyFAI's 2D histogram field dict. `count` is
/// `uint32`; `norm_sq` is the `(2, azim, rad)` doubleword array, flat C-order.
fn histresult2d_dict<'py>(
    py: Python<'py>,
    centers0: &[f64],
    centers1: &[f64],
    r: HistResult2d,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bins", (centers0.len(), centers1.len()))?;
    d.set_item("radial", centers0.to_vec().into_pyarray(py))?;
    d.set_item("azimuthal", centers1.to_vec().into_pyarray(py))?;
    d.set_item("intensity", r.intensity.into_pyarray(py))?;
    d.set_item("sigma", r.sigma.into_pyarray(py))?;
    d.set_item("std", r.std.into_pyarray(py))?;
    d.set_item("sem", r.sem.into_pyarray(py))?;
    d.set_item("count", r.count.into_pyarray(py))?;
    d.set_item("signal", r.signal.into_pyarray(py))?;
    d.set_item("variance", r.variance.into_pyarray(py))?;
    d.set_item("normalization", r.normalization.into_pyarray(py))?;
    d.set_item("norm_sq", r.norm_sq.into_pyarray(py))?;
    Ok(d)
}

#[pymethods]
impl GpuEngine {
    /// Cache a 1D CSR matrix on the GPU. `data`/`indices`/`indptr` are the parts
    /// `build_*_csr_1d` returns; `bin_centers` sets the radial axis (and the bin
    /// count). The image is integrated as `dtype` (default `-4` = int32, pyFAI's
    /// `_any2float` raw-int path). Corrections supplied here are detector-fixed
    /// and applied every frame on the GPU; `do_*` flags are derived from which
    /// arrays are given.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        data, indices, indptr, bin_centers, image_size, *,
        variance=None, dark=None, dark_variance=None, flat=None,
        solidangle=None, polarization=None, absorption=None, mask=None,
        error_model=0, dtype=-4, dummy=None, delta_dummy=0.0,
        normalization_factor=1.0, apply_normalization=false,
        empty=0.0, wg_min=32, prefer_gpu=true
    ))]
    fn from_csr_1d(
        data: PyReadonlyArray1<'_, f32>,
        indices: PyReadonlyArray1<'_, i32>,
        indptr: PyReadonlyArray1<'_, i32>,
        bin_centers: PyReadonlyArray1<'_, f64>,
        image_size: usize,
        variance: Option<PyReadonlyArray1<'_, f32>>,
        dark: Option<PyReadonlyArray1<'_, f32>>,
        dark_variance: Option<PyReadonlyArray1<'_, f32>>,
        flat: Option<PyReadonlyArray1<'_, f32>>,
        solidangle: Option<PyReadonlyArray1<'_, f32>>,
        polarization: Option<PyReadonlyArray1<'_, f32>>,
        absorption: Option<PyReadonlyArray1<'_, f32>>,
        mask: Option<PyReadonlyArray1<'_, i8>>,
        error_model: i32,
        dtype: i32,
        dummy: Option<f32>,
        delta_dummy: f32,
        normalization_factor: f32,
        apply_normalization: bool,
        empty: f32,
        wg_min: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let (arrays, corr) = corr_from_kwargs(
            variance,
            dark,
            dark_variance,
            flat,
            solidangle,
            polarization,
            absorption,
            mask,
            error_model,
            dtype,
            dummy,
            delta_dummy,
            normalization_factor,
            apply_normalization,
        )?;
        Self::build_csr(
            as_slice_1d(&data)?.to_vec(),
            as_slice_1d(&indices)?.to_vec(),
            as_slice_1d(&indptr)?.to_vec(),
            as_slice_1d(&bin_centers)?.to_vec(),
            None,
            image_size,
            corr,
            arrays,
            empty,
            wg_min,
            prefer_gpu,
        )
    }

    /// Cache a 2D CSR matrix on the GPU. Like [`from_csr_1d`](GpuEngine::from_csr_1d)
    /// but with radial + azimuthal bin centers; `bins == bins_rad * bins_azim`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        data, indices, indptr, bin_centers0, bin_centers1, image_size, *,
        variance=None, dark=None, dark_variance=None, flat=None,
        solidangle=None, polarization=None, absorption=None, mask=None,
        error_model=0, dtype=-4, dummy=None, delta_dummy=0.0,
        normalization_factor=1.0, apply_normalization=false,
        empty=0.0, wg_min=32, prefer_gpu=true
    ))]
    fn from_csr_2d(
        data: PyReadonlyArray1<'_, f32>,
        indices: PyReadonlyArray1<'_, i32>,
        indptr: PyReadonlyArray1<'_, i32>,
        bin_centers0: PyReadonlyArray1<'_, f64>,
        bin_centers1: PyReadonlyArray1<'_, f64>,
        image_size: usize,
        variance: Option<PyReadonlyArray1<'_, f32>>,
        dark: Option<PyReadonlyArray1<'_, f32>>,
        dark_variance: Option<PyReadonlyArray1<'_, f32>>,
        flat: Option<PyReadonlyArray1<'_, f32>>,
        solidangle: Option<PyReadonlyArray1<'_, f32>>,
        polarization: Option<PyReadonlyArray1<'_, f32>>,
        absorption: Option<PyReadonlyArray1<'_, f32>>,
        mask: Option<PyReadonlyArray1<'_, i8>>,
        error_model: i32,
        dtype: i32,
        dummy: Option<f32>,
        delta_dummy: f32,
        normalization_factor: f32,
        apply_normalization: bool,
        empty: f32,
        wg_min: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let (arrays, corr) = corr_from_kwargs(
            variance,
            dark,
            dark_variance,
            flat,
            solidangle,
            polarization,
            absorption,
            mask,
            error_model,
            dtype,
            dummy,
            delta_dummy,
            normalization_factor,
            apply_normalization,
        )?;
        Self::build_csr(
            as_slice_1d(&data)?.to_vec(),
            as_slice_1d(&indices)?.to_vec(),
            as_slice_1d(&indptr)?.to_vec(),
            as_slice_1d(&bin_centers0)?.to_vec(),
            Some(as_slice_1d(&bin_centers1)?.to_vec()),
            image_size,
            corr,
            arrays,
            empty,
            wg_min,
            prefer_gpu,
        )
    }

    /// Cache a 1D densified LUT on the GPU. `idx`/`coef` are the row-major
    /// `(bins, lut_size)` LUT (`build_*_lut_1d`'s parts, flattened); `bin_centers`
    /// sets the radial axis. The image is host-cast i32→f32 by the kernel path, so
    /// there is no `dtype` argument. Result dict matches the CSR 1D dict.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        idx, coef, bin_centers, image_size, lut_size, *,
        variance=None, dark=None, dark_variance=None, flat=None,
        solidangle=None, polarization=None, absorption=None, mask=None,
        error_model=0, dummy=None, delta_dummy=0.0,
        normalization_factor=1.0, apply_normalization=false,
        empty=0.0, block_size=32, prefer_gpu=true
    ))]
    fn from_lut_1d(
        idx: PyReadonlyArray1<'_, i32>,
        coef: PyReadonlyArray1<'_, f32>,
        bin_centers: PyReadonlyArray1<'_, f64>,
        image_size: usize,
        lut_size: usize,
        variance: Option<PyReadonlyArray1<'_, f32>>,
        dark: Option<PyReadonlyArray1<'_, f32>>,
        dark_variance: Option<PyReadonlyArray1<'_, f32>>,
        flat: Option<PyReadonlyArray1<'_, f32>>,
        solidangle: Option<PyReadonlyArray1<'_, f32>>,
        polarization: Option<PyReadonlyArray1<'_, f32>>,
        absorption: Option<PyReadonlyArray1<'_, f32>>,
        mask: Option<PyReadonlyArray1<'_, i8>>,
        error_model: i32,
        dummy: Option<f32>,
        delta_dummy: f32,
        normalization_factor: f32,
        apply_normalization: bool,
        empty: f32,
        block_size: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let (arrays, corr) = corr_from_kwargs(
            variance,
            dark,
            dark_variance,
            flat,
            solidangle,
            polarization,
            absorption,
            mask,
            error_model,
            -4,
            dummy,
            delta_dummy,
            normalization_factor,
            apply_normalization,
        )?;
        Self::build_lut(
            as_slice_1d(&idx)?.to_vec(),
            as_slice_1d(&coef)?.to_vec(),
            lut_size,
            as_slice_1d(&bin_centers)?.to_vec(),
            None,
            image_size,
            corr,
            arrays,
            empty,
            block_size,
            prefer_gpu,
        )
    }

    /// Cache a 2D densified LUT on the GPU. Like [`from_lut_1d`](GpuEngine::from_lut_1d)
    /// but with radial + azimuthal bin centers; `bins == bins_rad * bins_azim`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        idx, coef, bin_centers0, bin_centers1, image_size, lut_size, *,
        variance=None, dark=None, dark_variance=None, flat=None,
        solidangle=None, polarization=None, absorption=None, mask=None,
        error_model=0, dummy=None, delta_dummy=0.0,
        normalization_factor=1.0, apply_normalization=false,
        empty=0.0, block_size=32, prefer_gpu=true
    ))]
    fn from_lut_2d(
        idx: PyReadonlyArray1<'_, i32>,
        coef: PyReadonlyArray1<'_, f32>,
        bin_centers0: PyReadonlyArray1<'_, f64>,
        bin_centers1: PyReadonlyArray1<'_, f64>,
        image_size: usize,
        lut_size: usize,
        variance: Option<PyReadonlyArray1<'_, f32>>,
        dark: Option<PyReadonlyArray1<'_, f32>>,
        dark_variance: Option<PyReadonlyArray1<'_, f32>>,
        flat: Option<PyReadonlyArray1<'_, f32>>,
        solidangle: Option<PyReadonlyArray1<'_, f32>>,
        polarization: Option<PyReadonlyArray1<'_, f32>>,
        absorption: Option<PyReadonlyArray1<'_, f32>>,
        mask: Option<PyReadonlyArray1<'_, i8>>,
        error_model: i32,
        dummy: Option<f32>,
        delta_dummy: f32,
        normalization_factor: f32,
        apply_normalization: bool,
        empty: f32,
        block_size: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let (arrays, corr) = corr_from_kwargs(
            variance,
            dark,
            dark_variance,
            flat,
            solidangle,
            polarization,
            absorption,
            mask,
            error_model,
            -4,
            dummy,
            delta_dummy,
            normalization_factor,
            apply_normalization,
        )?;
        Self::build_lut(
            as_slice_1d(&idx)?.to_vec(),
            as_slice_1d(&coef)?.to_vec(),
            lut_size,
            as_slice_1d(&bin_centers0)?.to_vec(),
            Some(as_slice_1d(&bin_centers1)?.to_vec()),
            image_size,
            corr,
            arrays,
            empty,
            block_size,
            prefer_gpu,
        )
    }

    /// Cache a 1D histogram engine on the GPU. `radial`/`azimuthal` are the
    /// per-pixel position arrays (length `image_size`); `bin_centers` sets the
    /// radial axis. `radial_mini`/`radial_maxi` bound the histogram; azimuth is
    /// range-checked only when `check_azim` is true (then `azim_mini`/`azim_maxi`
    /// apply). The atomic scatter makes the float columns non-deterministic;
    /// `count` (uint32) stays bit-exact.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        radial, azimuthal, bin_centers, image_size,
        radial_mini, radial_maxi, *,
        check_azim=false, azim_mini=0.0, azim_maxi=0.0,
        variance=None, dark=None, dark_variance=None, flat=None,
        solidangle=None, polarization=None, absorption=None, mask=None,
        error_model=0, dummy=None, delta_dummy=0.0,
        normalization_factor=1.0, apply_normalization=false,
        empty=0.0, block_size=32, prefer_gpu=true
    ))]
    fn from_histogram_1d(
        radial: PyReadonlyArray1<'_, f32>,
        azimuthal: PyReadonlyArray1<'_, f32>,
        bin_centers: PyReadonlyArray1<'_, f64>,
        image_size: usize,
        radial_mini: f32,
        radial_maxi: f32,
        check_azim: bool,
        azim_mini: f32,
        azim_maxi: f32,
        variance: Option<PyReadonlyArray1<'_, f32>>,
        dark: Option<PyReadonlyArray1<'_, f32>>,
        dark_variance: Option<PyReadonlyArray1<'_, f32>>,
        flat: Option<PyReadonlyArray1<'_, f32>>,
        solidangle: Option<PyReadonlyArray1<'_, f32>>,
        polarization: Option<PyReadonlyArray1<'_, f32>>,
        absorption: Option<PyReadonlyArray1<'_, f32>>,
        mask: Option<PyReadonlyArray1<'_, i8>>,
        error_model: i32,
        dummy: Option<f32>,
        delta_dummy: f32,
        normalization_factor: f32,
        apply_normalization: bool,
        empty: f32,
        block_size: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let (arrays, corr) = corr_from_kwargs(
            variance,
            dark,
            dark_variance,
            flat,
            solidangle,
            polarization,
            absorption,
            mask,
            error_model,
            -4,
            dummy,
            delta_dummy,
            normalization_factor,
            apply_normalization,
        )?;
        Self::build_histogram(
            as_slice_1d(&radial)?.to_vec(),
            as_slice_1d(&azimuthal)?.to_vec(),
            radial_mini,
            radial_maxi,
            azim_mini,
            azim_maxi,
            if check_azim { 1 } else { 0 },
            as_slice_1d(&bin_centers)?.to_vec(),
            None,
            image_size,
            corr,
            arrays,
            empty,
            block_size,
            prefer_gpu,
        )
    }

    /// Cache a 2D histogram engine on the GPU. Like
    /// [`from_histogram_1d`](GpuEngine::from_histogram_1d) but with radial +
    /// azimuthal bin centers; the 2D kernel always range-checks azimuth using
    /// `azim_mini`/`azim_maxi` (there is no `check_azim` flag).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        radial, azimuthal, bin_centers0, bin_centers1, image_size,
        radial_mini, radial_maxi, azim_mini, azim_maxi, *,
        variance=None, dark=None, dark_variance=None, flat=None,
        solidangle=None, polarization=None, absorption=None, mask=None,
        error_model=0, dummy=None, delta_dummy=0.0,
        normalization_factor=1.0, apply_normalization=false,
        empty=0.0, block_size=32, prefer_gpu=true
    ))]
    fn from_histogram_2d(
        radial: PyReadonlyArray1<'_, f32>,
        azimuthal: PyReadonlyArray1<'_, f32>,
        bin_centers0: PyReadonlyArray1<'_, f64>,
        bin_centers1: PyReadonlyArray1<'_, f64>,
        image_size: usize,
        radial_mini: f32,
        radial_maxi: f32,
        azim_mini: f32,
        azim_maxi: f32,
        variance: Option<PyReadonlyArray1<'_, f32>>,
        dark: Option<PyReadonlyArray1<'_, f32>>,
        dark_variance: Option<PyReadonlyArray1<'_, f32>>,
        flat: Option<PyReadonlyArray1<'_, f32>>,
        solidangle: Option<PyReadonlyArray1<'_, f32>>,
        polarization: Option<PyReadonlyArray1<'_, f32>>,
        absorption: Option<PyReadonlyArray1<'_, f32>>,
        mask: Option<PyReadonlyArray1<'_, i8>>,
        error_model: i32,
        dummy: Option<f32>,
        delta_dummy: f32,
        normalization_factor: f32,
        apply_normalization: bool,
        empty: f32,
        block_size: usize,
        prefer_gpu: bool,
    ) -> PyResult<Self> {
        let (arrays, corr) = corr_from_kwargs(
            variance,
            dark,
            dark_variance,
            flat,
            solidangle,
            polarization,
            absorption,
            mask,
            error_model,
            -4,
            dummy,
            delta_dummy,
            normalization_factor,
            apply_normalization,
        )?;
        Self::build_histogram(
            as_slice_1d(&radial)?.to_vec(),
            as_slice_1d(&azimuthal)?.to_vec(),
            radial_mini,
            radial_maxi,
            azim_mini,
            azim_maxi,
            0,
            as_slice_1d(&bin_centers0)?.to_vec(),
            Some(as_slice_1d(&bin_centers1)?.to_vec()),
            image_size,
            corr,
            arrays,
            empty,
            block_size,
            prefer_gpu,
        )
    }

    /// Integrate one raw frame into the 1D radial bins on the GPU, reusing the
    /// cached program + matrix + corrections. `image` is the raw detector frame
    /// (int32, length `image_size`). Returns the field dict for the engine's algo
    /// (CSR/LUT: `position`/`intensity`/`sigma`/`sum_signal`/`sum_variance`/
    /// `sum_normalization`/`count`/`std`/`sem`/`sum_norm_sq`; histogram omits
    /// `std`/`sem`/`sum_norm_sq`). Raises `ValueError` on a 2D engine.
    fn integrate1d<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.centers1.is_some() {
            return Err(PyValueError::new_err("2D engine: call integrate2d"));
        }
        let img = self.check_image(&image)?;
        match &self.matrix {
            GpuMatrix::Csr {
                data,
                indices,
                indptr,
                wg_min,
            } => {
                let inputs = self.csr_inputs(img, data, indices, indptr);
                let r = ocl_err(
                    integrate1d_csr(
                        &self.session(),
                        &inputs,
                        &self.corr.to_csr_args(),
                        self.empty,
                        *wg_min,
                    ),
                    "integrate1d_csr",
                )?;
                result1d_dict(py, &self.centers0, r)
            }
            GpuMatrix::Lut {
                idx,
                coef,
                lut_size,
                block_size,
            } => {
                let inputs = self.lut_inputs(img, idx, coef, *lut_size, self.centers0.len());
                let r = ocl_err(
                    integrate1d_lut(
                        &self.session(),
                        &inputs,
                        &self.corr.to_corr4_args(),
                        self.empty,
                        *block_size,
                    ),
                    "integrate1d_lut",
                )?;
                result1d_dict(py, &self.centers0, r)
            }
            GpuMatrix::Histogram {
                radial,
                azimuthal,
                radial_mini,
                radial_maxi,
                azim_mini,
                azim_maxi,
                check_azim,
                block_size,
            } => {
                let inputs = self.hist_inputs(img, radial, azimuthal, self.centers0.len());
                let scalars = HistogramScalars {
                    radial_mini: *radial_mini,
                    radial_maxi: *radial_maxi,
                    check_azim: *check_azim,
                    azim_mini: *azim_mini,
                    azim_maxi: *azim_maxi,
                    empty: self.empty,
                };
                let r = ocl_err(
                    integrate1d_histogram(
                        &self.session(),
                        &inputs,
                        &self.corr.to_corr4_args(),
                        &scalars,
                        *block_size,
                    ),
                    "integrate1d_histogram",
                )?;
                histresult1d_dict(py, &self.centers0, r)
            }
        }
    }

    /// Integrate one raw frame into the 2D (azimuthal, radial) map on the GPU.
    /// Returns the algo's field dict (`bins`/`radial`/`azimuthal` plus the 2D
    /// columns), flat (azimuthal, radial) C-order. Raises `ValueError` on a 1D
    /// engine.
    fn integrate2d<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let Some(centers1) = self.centers1.as_ref() else {
            return Err(PyValueError::new_err("1D engine: call integrate1d"));
        };
        let img = self.check_image(&image)?;
        let bins_rad = self.centers0.len();
        let bins_azim = centers1.len();
        let bins = bins_rad * bins_azim;
        match &self.matrix {
            GpuMatrix::Csr {
                data,
                indices,
                indptr,
                wg_min,
            } => {
                let inputs = self.csr_inputs(img, data, indices, indptr);
                let r = ocl_err(
                    integrate2d_csr(
                        &self.session(),
                        &inputs,
                        &self.corr.to_csr_args(),
                        self.empty,
                        *wg_min,
                        bins_rad,
                        bins_azim,
                    ),
                    "integrate2d_csr",
                )?;
                result2d_dict(py, &self.centers0, centers1, r)
            }
            GpuMatrix::Lut {
                idx,
                coef,
                lut_size,
                block_size,
            } => {
                let inputs = self.lut_inputs(img, idx, coef, *lut_size, bins);
                let r = ocl_err(
                    integrate2d_lut(
                        &self.session(),
                        &inputs,
                        &self.corr.to_corr4_args(),
                        self.empty,
                        *block_size,
                        bins_rad,
                        bins_azim,
                    ),
                    "integrate2d_lut",
                )?;
                result2d_dict(py, &self.centers0, centers1, r)
            }
            GpuMatrix::Histogram {
                radial,
                azimuthal,
                radial_mini,
                radial_maxi,
                azim_mini,
                azim_maxi,
                check_azim,
                block_size,
            } => {
                let inputs = self.hist_inputs(img, radial, azimuthal, bins);
                let scalars = HistogramScalars {
                    radial_mini: *radial_mini,
                    radial_maxi: *radial_maxi,
                    check_azim: *check_azim,
                    azim_mini: *azim_mini,
                    azim_maxi: *azim_maxi,
                    empty: self.empty,
                };
                let r = ocl_err(
                    integrate2d_histogram(
                        &self.session(),
                        &inputs,
                        &self.corr.to_corr4_args(),
                        &scalars,
                        *block_size,
                        bins_rad,
                        bins_azim,
                    ),
                    "integrate2d_histogram",
                )?;
                histresult2d_dict(py, &self.centers0, centers1, r)
            }
        }
    }
}
