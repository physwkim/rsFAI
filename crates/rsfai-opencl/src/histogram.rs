//! Host-side orchestration of pyFAI's histogram-NG OpenCL pipeline.
//!
//! Reproduces `OCL_Histogram1d`/`OCL_Histogram2d.integrate`
//! (`pyFAI/opencl/azim_hist.py`): `memset_histograms` → `corrections4` →
//! `histogram_{1d,2d}_preproc` → `histogram_postproc`, driving pyFAI's own
//! embedded kernels (see [`crate::program::build_histogram_program`]). As with
//! the CSR/LUT backends the arithmetic lives in the GPU kernels; the Rust side
//! only uploads, sets arguments in pyFAI's order, and reads back.
//!
//! Unlike CSR/LUT this path has **no sparse matrix**. Each pixel's preprocessed
//! `float4` is scattered into the bins with `atomic_add` (the bin is computed
//! from the per-pixel `radial`/`azimuthal` position arrays). The atomic commit
//! order across work-items is **not deterministic**, so this backend is never
//! bit-exact — pyFAI is not even reproducible against itself here (measured
//! intensity self-divergence ~3.4e-6 in 1D, ~7.0e-6 in 2D over 8 runs). It is
//! validated against a looser, measured atomic-noise gate; `count` (integer
//! atomics) stays bit-exact regardless. See the test's `HIST_REL_TOL`.
//!
//! On the Apple M4 Pro (no `cl_khr_int64_base_atomics`) the doubleword Kahan
//! atomic degrades to a 32-bit `atomic_cmpxchg` on the high word only (the
//! kernel's `#else` path): the low word of every `(hi, lo)` histogram pair stays
//! zero, so the accumulation is effectively a plain `f32` atomic sum.
//!
//! The image pre-cast (`i32 as f32`) and the `corrections4` argument order are
//! identical to [`crate::lut`]; that kernel and the `Corrections4Args` struct
//! are shared.

use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_char, cl_float, cl_uint, CL_BLOCKING};
use std::ptr;

use crate::common::{transpose_to_azim_rad, upload_f32, ClSession};
use crate::lut::Corrections4Args;

/// Per-pixel inputs to the histogram pipeline. The image is raw `i32` (host-cast
/// to `f32` on upload); correction arrays are `f32` (caller casts f64→f32 like
/// pyFAI's `send_buffer`). `radial`/`azimuthal` are the per-pixel position
/// arrays the histogram bins from (length = image size, `f32` like the GPU
/// buffers).
pub struct HistogramInputs<'a> {
    /// Raw image as i32. Length = image size.
    pub image_i32: &'a [i32],
    pub variance: Option<&'a [f32]>,
    pub dark: Option<&'a [f32]>,
    pub dark_variance: Option<&'a [f32]>,
    pub flat: Option<&'a [f32]>,
    pub solidangle: Option<&'a [f32]>,
    pub polarization: Option<&'a [f32]>,
    pub absorption: Option<&'a [f32]>,
    pub mask: Option<&'a [i8]>,
    /// Per-pixel radial position (the histogram axis). Length = image size.
    pub radial: &'a [f32],
    /// Per-pixel azimuthal position. Length = image size. Read by the kernel only
    /// when `check_azim != 0` (1D) or always (2D).
    pub azimuthal: &'a [f32],
    /// Total number of bins (for 2D this is `bins_rad * bins_azim`).
    pub bins: usize,
}

/// The histogram-preproc range scalars, captured from pyFAI's per-call
/// `cl_kernel_args`. `check_azim` applies only to the 1D kernel (the 2D kernel
/// has no such flag — it always range-checks azimuth using `azim_mini/maxi`).
#[derive(Debug, Clone, Copy)]
pub struct HistogramScalars {
    pub radial_mini: f32,
    pub radial_maxi: f32,
    pub check_azim: i8,
    pub azim_mini: f32,
    pub azim_maxi: f32,
    pub empty: f32,
}

/// Which preproc kernel to enqueue, and the 2D cell geometry when applicable.
enum HistKind {
    OneD,
    TwoD { bins_rad: usize, bins_azim: usize },
}

/// Raw read-back of the histogram pipeline. `histo_*` are flat `(bins, 2)`
/// doubleword pairs (`[hi, lo]` per bin); `count` is one `u32` per bin.
struct HistogramOutput {
    intensity: Vec<f32>,
    std: Vec<f32>,
    sem: Vec<f32>,
    histo_sig: Vec<f32>,
    histo_var: Vec<f32>,
    histo_nrm: Vec<f32>,
    histo_nrm2: Vec<f32>,
    count: Vec<u32>,
}

/// 1D histogram result, matching pyFAI's `OCL_Histogram1d` `Integrate1dtpl`
/// mapping. `intensity == averint`, `sigma == sem`. `signal`/`variance`/
/// `normalization` are the flat `(bins, 2)` doubleword histograms in row-major
/// `[bin][hi|lo]` order (on the M4 Pro the `lo` lane is always 0). `std`/`sem`/
/// `norm_sq` are **not** exposed by pyFAI's 1D result for this path.
#[derive(Debug, Clone)]
pub struct HistResult1d {
    pub intensity: Vec<f32>,
    pub sigma: Vec<f32>,
    pub count: Vec<u32>,
    pub signal: Vec<f32>,
    pub variance: Vec<f32>,
    pub normalization: Vec<f32>,
}

/// 2D histogram result. Every field is `(azim, rad)` C-order flat
/// (`field[azim * bins_rad + rad]` = pyFAI's `field.T`) except `norm_sq`, which
/// is `(2, azim, rad)` C-order (pyFAI's `histo_nrm2.T` of the `(rad, azim, 2)`
/// doubleword array). `signal`/`variance`/`normalization` use the **high**
/// doubleword lane only (pyFAI's `histo_*[:,:,0].T`).
#[derive(Debug, Clone)]
pub struct HistResult2d {
    pub intensity: Vec<f32>,
    pub sigma: Vec<f32>,
    pub std: Vec<f32>,
    pub sem: Vec<f32>,
    pub count: Vec<u32>,
    pub signal: Vec<f32>,
    pub variance: Vec<f32>,
    pub normalization: Vec<f32>,
    pub norm_sq: Vec<f32>,
}

/// Run pyFAI's histogram-NG OpenCL pipeline on `program` (built by
/// [`crate::program::build_histogram_program`] for the matching bins / image
/// size / work-group size) and read back the flat per-bin buffers. `block_size`
/// is pyFAI's `BLOCK_SIZE` (32 on the M4 Pro).
fn run_histogram_pipeline(
    session: &ClSession<'_>,
    inputs: &HistogramInputs<'_>,
    corr: &Corrections4Args,
    scalars: &HistogramScalars,
    block_size: usize,
    kind: &HistKind,
) -> Result<HistogramOutput, String> {
    let context = session.context;
    let queue = session.queue;
    let program = session.program;
    let size = inputs.image_i32.len();
    let bins = inputs.bins;
    assert_eq!(inputs.radial.len(), size, "radial length != image size");
    assert_eq!(
        inputs.azimuthal.len(),
        size,
        "azimuthal length != image size"
    );

    // ---- Upload inputs ------------------------------------------------
    // Image is pre-cast to float (pyFAI's s32_to_float); `i32 as f32` matches.
    let image_host: Vec<cl_float> = inputs.image_i32.iter().map(|&v| v as f32).collect();
    let mut image_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_ONLY, size, ptr::null_mut())
            .map_err(|e| format!("alloc image: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut image_buf, CL_BLOCKING, 0, &image_host, &[]) }
        .map_err(|e| format!("write image: {e}"))?;

    let variance_buf = upload_f32(context, queue, inputs.variance, size)?;
    let dark_buf = upload_f32(context, queue, inputs.dark, size)?;
    let dark_variance_buf = upload_f32(context, queue, inputs.dark_variance, size)?;
    let flat_buf = upload_f32(context, queue, inputs.flat, size)?;
    let solidangle_buf = upload_f32(context, queue, inputs.solidangle, size)?;
    let polarization_buf = upload_f32(context, queue, inputs.polarization, size)?;
    let absorption_buf = upload_f32(context, queue, inputs.absorption, size)?;
    let radial_buf = upload_f32(context, queue, Some(inputs.radial), size)?;
    let azimuthal_buf = upload_f32(context, queue, Some(inputs.azimuthal), size)?;

    let mask_host: Vec<cl_char> = match inputs.mask {
        Some(m) => m.to_vec(),
        None => vec![0i8; size],
    };
    let mut mask_buf = unsafe {
        Buffer::<cl_char>::create(context, CL_MEM_READ_ONLY, size, ptr::null_mut())
            .map_err(|e| format!("alloc mask: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut mask_buf, CL_BLOCKING, 0, &mask_host, &[]) }
        .map_err(|e| format!("write mask: {e}"))?;

    // ---- Allocate outputs ---------------------------------------------
    // histo_* are float2 (doubleword) buffers: 2 floats per bin. histo_cnt is one
    // uint per bin. output4 is the per-pixel float4 corrections result.
    let output4_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, 4 * size, ptr::null_mut())
            .map_err(|e| format!("alloc output4: {e}"))?
    };
    let new_dw = |label: &str| unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, 2 * bins, ptr::null_mut())
            .map_err(|e| format!("alloc {label}: {e}"))
    };
    let histo_sig_buf = new_dw("histo_sig")?;
    let histo_var_buf = new_dw("histo_var")?;
    let histo_nrm_buf = new_dw("histo_nrm")?;
    let histo_nrm2_buf = new_dw("histo_nrm2")?;
    let histo_cnt_buf = unsafe {
        Buffer::<cl_uint>::create(context, CL_MEM_READ_WRITE, bins, ptr::null_mut())
            .map_err(|e| format!("alloc histo_cnt: {e}"))?
    };
    let intensity_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_WRITE_ONLY, bins, ptr::null_mut())
            .map_err(|e| format!("alloc intensity: {e}"))?
    };
    let std_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_WRITE_ONLY, bins, ptr::null_mut())
            .map_err(|e| format!("alloc std: {e}"))?
    };
    let sem_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_WRITE_ONLY, bins, ptr::null_mut())
            .map_err(|e| format!("alloc sem: {e}"))?
    };

    let g_bins = bins.div_ceil(block_size) * block_size;
    let g_data = size.div_ceil(block_size) * block_size;

    // ---- memset_histograms: zero histo_sig/var/nrm/nrm2/cnt ----------
    let memset = Kernel::create(program, "memset_histograms")
        .map_err(|e| format!("memset_histograms: {e}"))?;
    unsafe {
        ExecuteKernel::new(&memset)
            .set_arg(&histo_sig_buf)
            .set_arg(&histo_var_buf)
            .set_arg(&histo_nrm_buf)
            .set_arg(&histo_nrm2_buf)
            .set_arg(&histo_cnt_buf)
            .set_arg(&(bins as cl_uint))
            .set_global_work_size(g_bins)
            .set_local_work_size(block_size)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue memset_histograms: {e}"))?;
    }

    // ---- corrections4: per-pixel preprocessing into output4 (no dtype) ----
    let corrections =
        Kernel::create(program, "corrections4").map_err(|e| format!("corrections4: {e}"))?;
    unsafe {
        ExecuteKernel::new(&corrections)
            .set_arg(&image_buf)
            .set_arg(&(corr.error_model as cl_char))
            .set_arg(&variance_buf)
            .set_arg(&(corr.do_dark as cl_char))
            .set_arg(&dark_buf)
            .set_arg(&(corr.do_dark_variance as cl_char))
            .set_arg(&dark_variance_buf)
            .set_arg(&(corr.do_flat as cl_char))
            .set_arg(&flat_buf)
            .set_arg(&(corr.do_solidangle as cl_char))
            .set_arg(&solidangle_buf)
            .set_arg(&(corr.do_polarization as cl_char))
            .set_arg(&polarization_buf)
            .set_arg(&(corr.do_absorption as cl_char))
            .set_arg(&absorption_buf)
            .set_arg(&(corr.do_mask as cl_char))
            .set_arg(&mask_buf)
            .set_arg(&(corr.do_dummy as cl_char))
            .set_arg(&(corr.dummy as cl_float))
            .set_arg(&(corr.delta_dummy as cl_float))
            .set_arg(&(corr.normalization_factor as cl_float))
            .set_arg(&(corr.apply_normalization as cl_char))
            .set_arg(&output4_buf)
            .set_global_work_size(g_data)
            .set_local_work_size(block_size)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue corrections4: {e}"))?;
    }

    // ---- histogram_{1d,2d}_preproc: atomic-add scatter into the bins ----
    match *kind {
        HistKind::OneD => {
            let hist = Kernel::create(program, "histogram_1d_preproc")
                .map_err(|e| format!("histogram_1d_preproc: {e}"))?;
            unsafe {
                ExecuteKernel::new(&hist)
                    .set_arg(&radial_buf)
                    .set_arg(&azimuthal_buf)
                    .set_arg(&output4_buf)
                    .set_arg(&histo_sig_buf)
                    .set_arg(&histo_var_buf)
                    .set_arg(&histo_nrm_buf)
                    .set_arg(&histo_nrm2_buf)
                    .set_arg(&histo_cnt_buf)
                    .set_arg(&(size as cl_uint))
                    .set_arg(&(bins as cl_uint))
                    .set_arg(&(scalars.radial_mini as cl_float))
                    .set_arg(&(scalars.radial_maxi as cl_float))
                    .set_arg(&(scalars.check_azim as cl_char))
                    .set_arg(&(scalars.azim_mini as cl_float))
                    .set_arg(&(scalars.azim_maxi as cl_float))
                    .set_global_work_size(g_data)
                    .set_local_work_size(block_size)
                    .enqueue_nd_range(queue)
                    .map_err(|e| format!("enqueue histogram_1d_preproc: {e}"))?;
            }
        }
        HistKind::TwoD {
            bins_rad,
            bins_azim,
        } => {
            let hist = Kernel::create(program, "histogram_2d_preproc")
                .map_err(|e| format!("histogram_2d_preproc: {e}"))?;
            unsafe {
                ExecuteKernel::new(&hist)
                    .set_arg(&radial_buf)
                    .set_arg(&azimuthal_buf)
                    .set_arg(&output4_buf)
                    .set_arg(&histo_sig_buf)
                    .set_arg(&histo_var_buf)
                    .set_arg(&histo_nrm_buf)
                    .set_arg(&histo_nrm2_buf)
                    .set_arg(&histo_cnt_buf)
                    .set_arg(&(size as cl_uint))
                    .set_arg(&(bins_rad as cl_uint))
                    .set_arg(&(bins_azim as cl_uint))
                    .set_arg(&(scalars.radial_mini as cl_float))
                    .set_arg(&(scalars.radial_maxi as cl_float))
                    .set_arg(&(scalars.azim_mini as cl_float))
                    .set_arg(&(scalars.azim_maxi as cl_float))
                    .set_global_work_size(g_data)
                    .set_local_work_size(block_size)
                    .enqueue_nd_range(queue)
                    .map_err(|e| format!("enqueue histogram_2d_preproc: {e}"))?;
            }
        }
    }

    // ---- histogram_postproc: doubleword histograms → intensity/std/sem ----
    let postproc = Kernel::create(program, "histogram_postproc")
        .map_err(|e| format!("histogram_postproc: {e}"))?;
    unsafe {
        ExecuteKernel::new(&postproc)
            .set_arg(&histo_sig_buf)
            .set_arg(&histo_var_buf)
            .set_arg(&histo_nrm_buf)
            .set_arg(&histo_nrm2_buf)
            .set_arg(&histo_cnt_buf)
            .set_arg(&(bins as cl_uint))
            .set_arg(&(scalars.empty as cl_float))
            .set_arg(&intensity_buf)
            .set_arg(&std_buf)
            .set_arg(&sem_buf)
            .set_global_work_size(g_bins)
            .set_local_work_size(block_size)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue histogram_postproc: {e}"))?;
    }

    // ---- Read back ----------------------------------------------------
    let mut intensity = vec![0.0f32; bins];
    let mut std = vec![0.0f32; bins];
    let mut sem = vec![0.0f32; bins];
    let mut histo_sig = vec![0.0f32; 2 * bins];
    let mut histo_var = vec![0.0f32; 2 * bins];
    let mut histo_nrm = vec![0.0f32; 2 * bins];
    let mut histo_nrm2 = vec![0.0f32; 2 * bins];
    let mut count = vec![0u32; bins];
    unsafe {
        queue
            .enqueue_read_buffer(&intensity_buf, CL_BLOCKING, 0, &mut intensity, &[])
            .map_err(|e| format!("read intensity: {e}"))?;
        queue
            .enqueue_read_buffer(&std_buf, CL_BLOCKING, 0, &mut std, &[])
            .map_err(|e| format!("read std: {e}"))?;
        queue
            .enqueue_read_buffer(&sem_buf, CL_BLOCKING, 0, &mut sem, &[])
            .map_err(|e| format!("read sem: {e}"))?;
        queue
            .enqueue_read_buffer(&histo_sig_buf, CL_BLOCKING, 0, &mut histo_sig, &[])
            .map_err(|e| format!("read histo_sig: {e}"))?;
        queue
            .enqueue_read_buffer(&histo_var_buf, CL_BLOCKING, 0, &mut histo_var, &[])
            .map_err(|e| format!("read histo_var: {e}"))?;
        queue
            .enqueue_read_buffer(&histo_nrm_buf, CL_BLOCKING, 0, &mut histo_nrm, &[])
            .map_err(|e| format!("read histo_nrm: {e}"))?;
        queue
            .enqueue_read_buffer(&histo_nrm2_buf, CL_BLOCKING, 0, &mut histo_nrm2, &[])
            .map_err(|e| format!("read histo_nrm2: {e}"))?;
        queue
            .enqueue_read_buffer(&histo_cnt_buf, CL_BLOCKING, 0, &mut count, &[])
            .map_err(|e| format!("read histo_cnt: {e}"))?;
    }

    Ok(HistogramOutput {
        intensity,
        std,
        sem,
        histo_sig,
        histo_var,
        histo_nrm,
        histo_nrm2,
        count,
    })
}

/// Integrate to a 1D radial curve via the histogram-NG pipeline.
pub fn integrate1d_histogram(
    session: &ClSession<'_>,
    inputs: &HistogramInputs<'_>,
    corr: &Corrections4Args,
    scalars: &HistogramScalars,
    block_size: usize,
) -> Result<HistResult1d, String> {
    let out = run_histogram_pipeline(session, inputs, corr, scalars, block_size, &HistKind::OneD)?;
    Ok(HistResult1d {
        intensity: out.intensity,
        sigma: out.sem,
        count: out.count,
        signal: out.histo_sig,
        variance: out.histo_var,
        normalization: out.histo_nrm,
    })
}

/// Take the high doubleword lane of a flat `(bins, 2)` histogram: `hi[b] =
/// histo[2*b]`.
fn high_lane(histo: &[f32], bins: usize) -> Vec<f32> {
    (0..bins).map(|b| histo[2 * b]).collect()
}

/// Transpose a flat radial-major `u32` field into `(azim, rad)` C-order, like
/// [`transpose_to_azim_rad`] for `f32`.
fn transpose_u32_to_azim_rad(v: &[u32], bins_rad: usize, bins_azim: usize) -> Vec<u32> {
    let mut out = vec![0u32; bins_rad * bins_azim];
    for rad in 0..bins_rad {
        for azim in 0..bins_azim {
            out[azim * bins_rad + rad] = v[rad * bins_azim + azim];
        }
    }
    out
}

/// Integrate to a 2D `(azim, rad)` map via the histogram-NG pipeline. The 2D
/// kernel bins radial-major (`cell = rad * bins_azim + azim`); the result is
/// pyFAI's per-field transpose (see [`HistResult2d`]).
pub fn integrate2d_histogram(
    session: &ClSession<'_>,
    inputs: &HistogramInputs<'_>,
    corr: &Corrections4Args,
    scalars: &HistogramScalars,
    block_size: usize,
    bins_rad: usize,
    bins_azim: usize,
) -> Result<HistResult2d, String> {
    let out = run_histogram_pipeline(
        session,
        inputs,
        corr,
        scalars,
        block_size,
        &HistKind::TwoD {
            bins_rad,
            bins_azim,
        },
    )?;
    let bins = bins_rad * bins_azim;
    let t = |v: &[f32]| transpose_to_azim_rad(v, bins_rad, bins_azim);
    // norm_sq is pyFAI's histo_nrm2.T: the (rad, azim, 2) doubleword array
    // transposed to (2, azim, rad), C-order: out[k, azim, rad] = nrm2[rad, azim, k].
    let mut norm_sq = vec![0.0f32; 2 * bins];
    for rad in 0..bins_rad {
        for azim in 0..bins_azim {
            for k in 0..2 {
                norm_sq[k * bins + azim * bins_rad + rad] =
                    out.histo_nrm2[2 * (rad * bins_azim + azim) + k];
            }
        }
    }
    Ok(HistResult2d {
        intensity: t(&out.intensity),
        sigma: t(&out.sem),
        std: t(&out.std),
        sem: t(&out.sem),
        count: transpose_u32_to_azim_rad(&out.count, bins_rad, bins_azim),
        signal: t(&high_lane(&out.histo_sig, bins)),
        variance: t(&high_lane(&out.histo_var, bins)),
        normalization: t(&high_lane(&out.histo_nrm, bins)),
        norm_sq,
    })
}
