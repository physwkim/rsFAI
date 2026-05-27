//! Host-side orchestration of pyFAI's LUT-NG OpenCL pipeline.
//!
//! Reproduces `OCL_LUT_Integrator.integrate_ng` (`pyFAI/opencl/azim_lut.py`):
//! `memset_out` → `corrections4` → `lut_integrate4`, driving pyFAI's own
//! embedded kernels (see [`crate::program::build_lut_program`]). Like the CSR
//! backend the arithmetic lives entirely in the GPU kernels; the Rust side only
//! uploads buffers, sets arguments in pyFAI's exact order, and reads back.
//!
//! Two things differ from the CSR path:
//!
//! * **Image is pre-cast to `float`.** pyFAI's `send_buffer` runs a
//!   `s32_to_float` kernel into the `float` `image` buffer, so `corrections4`
//!   reads `global float*` and takes **no** `dtype` argument. We host-cast
//!   `i32 as f32` (IEEE round-to-nearest-even, identical to OpenCL's `(float)int`
//!   and numpy's `astype`) and upload to a float buffer.
//! * **LUT is a densified, transposed sparse matrix.** The look-up table is a
//!   `(bins, lut_size)` array of `struct lut_point_t { int idx; float coef; }`;
//!   on the GPU (`ON_CPU == 0`) pyFAI uploads `lut.T` so the kernel reads
//!   `lut[j * NBINS + bin]` for coalesced access. `lut_integrate4` is one thread
//!   per bin (no work-group tree reduction), looping over `NLUT` entries with
//!   Kahan summation.
//!
//! The 1D/2D split is purely host-side packaging (see [`crate::common`]).

use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_char, cl_float, cl_int, CL_BLOCKING};
use std::ptr;

use crate::common::{pack_1d, pack_2d, upload_f32, ClSession, PipelineOutput, Result1d, Result2d};

/// The 13 scalar arguments of the `corrections4` preprocessing kernel, captured
/// verbatim from pyFAI's live `cl_kernel_args`. Identical to the CSR
/// `Corrections4aArgs` except there is **no `dtype`**: the LUT path pre-casts the
/// image to `float`, so `corrections4` reads `global float*`.
#[derive(Debug, Clone, Copy)]
pub struct Corrections4Args {
    /// `ErrorModel` enum value: 0 none, 1 variance, 2 poisson, 3 azimuthal.
    pub error_model: i8,
    pub do_dark: i8,
    pub do_dark_variance: i8,
    pub do_flat: i8,
    pub do_solidangle: i8,
    pub do_polarization: i8,
    pub do_absorption: i8,
    pub do_mask: i8,
    pub do_dummy: i8,
    pub dummy: f32,
    pub delta_dummy: f32,
    pub normalization_factor: f32,
    pub apply_normalization: i8,
}

/// A single densified-LUT entry, matching `struct lut_point_t { int idx; float
/// coef; }` in `ocl_azim_LUT.cl` (8 bytes, `idx` at offset 0, `coef` at 4).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LutPoint {
    idx: cl_int,
    coef: cl_float,
}

/// Per-pixel inputs to the LUT pipeline. The image is raw `i32` (host-cast to
/// `f32` on upload); correction arrays are `f32` (caller casts f64→f32 like
/// pyFAI's `send_buffer`). The LUT is the densified `(bins, lut_size)` matrix in
/// **row-major** order (`idx[bin * lut_size + j]`); this code transposes it for
/// the GPU.
pub struct LutInputs<'a> {
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
    /// Densified LUT indices, row-major `(bins, lut_size)`: `idx[bin*lut_size+j]`.
    pub lut_idx: &'a [i32],
    /// Densified LUT coefficients, row-major `(bins, lut_size)`.
    pub lut_coef: &'a [f32],
    /// Number of output bins (LUT rows). For 2D this is the flattened cell count.
    pub bins: usize,
    /// Densified LUT width (max non-zero entries per bin; pyFAI's `lut_size`).
    pub lut_size: usize,
}

/// Build the transposed device LUT pyFAI uploads on the GPU (`lut.T.copy()`):
/// `gpu[j * bins + bin] = lut[bin * lut_size + j]`, so the kernel's
/// `lut[j * NBINS + bin]` reads entry `j` of `bin`.
fn build_gpu_lut(inputs: &LutInputs<'_>) -> Vec<LutPoint> {
    let (bins, lut_size) = (inputs.bins, inputs.lut_size);
    let mut lut = Vec::with_capacity(bins * lut_size);
    for j in 0..lut_size {
        for bin in 0..bins {
            let src = bin * lut_size + j;
            lut.push(LutPoint {
                idx: inputs.lut_idx[src],
                coef: inputs.lut_coef[src],
            });
        }
    }
    lut
}

/// Run pyFAI's LUT-NG OpenCL pipeline on `program` (built by
/// [`crate::program::build_lut_program`] for the matching bins / image size /
/// `lut_size`) and read back the flat per-bin result. `block_size` is pyFAI's
/// `BLOCK_SIZE` (the work-group size, 32 on the M4 Pro).
fn run_lut_pipeline(
    session: &ClSession<'_>,
    inputs: &LutInputs<'_>,
    corr: &Corrections4Args,
    empty: f32,
    block_size: usize,
) -> Result<PipelineOutput, String> {
    let context = session.context;
    let queue = session.queue;
    let program = session.program;
    let size = inputs.image_i32.len();
    let bins = inputs.bins;
    let lut_size = inputs.lut_size;
    assert_eq!(
        inputs.lut_idx.len(),
        bins * lut_size,
        "lut_idx length != bins * lut_size"
    );
    assert_eq!(
        inputs.lut_coef.len(),
        bins * lut_size,
        "lut_coef length != bins * lut_size"
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

    let lut_host = build_gpu_lut(inputs);
    let mut lut_buf = unsafe {
        Buffer::<LutPoint>::create(context, CL_MEM_READ_ONLY, lut_host.len(), ptr::null_mut())
            .map_err(|e| format!("alloc lut: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, &lut_host, &[]) }
        .map_err(|e| format!("write lut: {e}"))?;

    // ---- Allocate outputs ---------------------------------------------
    let output4_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, 4 * size, ptr::null_mut())
            .map_err(|e| format!("alloc output4: {e}"))?
    };
    let merged8_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_WRITE_ONLY, 8 * bins, ptr::null_mut())
            .map_err(|e| format!("alloc merged8: {e}"))?
    };
    let averint_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, bins, ptr::null_mut())
            .map_err(|e| format!("alloc averint: {e}"))?
    };
    let std_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, bins, ptr::null_mut())
            .map_err(|e| format!("alloc std: {e}"))?
    };
    let sem_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, bins, ptr::null_mut())
            .map_err(|e| format!("alloc sem: {e}"))?
    };

    let g_bins = bins.div_ceil(block_size) * block_size;
    let g_data = size.div_ceil(block_size) * block_size;

    // ---- memset_out: zero averint, sem, merged8 ----------------------
    // (lut_integrate4 writes every bin, so this only clears padding; run it for
    // fidelity with integrate_ng. pyFAI uses the memset_out kernel here.)
    let memset = Kernel::create(program, "memset_out").map_err(|e| format!("memset_out: {e}"))?;
    unsafe {
        ExecuteKernel::new(&memset)
            .set_arg(&averint_buf)
            .set_arg(&sem_buf)
            .set_arg(&merged8_buf)
            .set_global_work_size(g_bins)
            .set_local_work_size(block_size)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue memset_out: {e}"))?;
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

    // ---- lut_integrate4: one thread per bin, loop over NLUT entries -------
    let integrate =
        Kernel::create(program, "lut_integrate4").map_err(|e| format!("lut_integrate4: {e}"))?;
    unsafe {
        ExecuteKernel::new(&integrate)
            .set_arg(&output4_buf)
            .set_arg(&lut_buf)
            .set_arg(&(empty as cl_float))
            .set_arg(&merged8_buf)
            .set_arg(&averint_buf)
            .set_arg(&std_buf)
            .set_arg(&sem_buf)
            .set_global_work_size(g_bins)
            .set_local_work_size(block_size)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue lut_integrate4: {e}"))?;
    }

    // ---- Read back ----------------------------------------------------
    let mut intensity = vec![0.0f32; bins];
    let mut std = vec![0.0f32; bins];
    let mut sem = vec![0.0f32; bins];
    let mut merged8 = vec![0.0f32; 8 * bins];
    unsafe {
        queue
            .enqueue_read_buffer(&averint_buf, CL_BLOCKING, 0, &mut intensity, &[])
            .map_err(|e| format!("read averint: {e}"))?;
        queue
            .enqueue_read_buffer(&std_buf, CL_BLOCKING, 0, &mut std, &[])
            .map_err(|e| format!("read std: {e}"))?;
        queue
            .enqueue_read_buffer(&sem_buf, CL_BLOCKING, 0, &mut sem, &[])
            .map_err(|e| format!("read sem: {e}"))?;
        queue
            .enqueue_read_buffer(&merged8_buf, CL_BLOCKING, 0, &mut merged8, &[])
            .map_err(|e| format!("read merged8: {e}"))?;
    }

    Ok(PipelineOutput {
        intensity,
        std,
        sem,
        merged8,
    })
}

/// Integrate to a 1D radial curve via the LUT-NG pipeline.
pub fn integrate1d_lut(
    session: &ClSession<'_>,
    inputs: &LutInputs<'_>,
    corr: &Corrections4Args,
    empty: f32,
    block_size: usize,
) -> Result<Result1d, String> {
    Ok(pack_1d(run_lut_pipeline(
        session, inputs, corr, empty, block_size,
    )?))
}

/// Integrate to a 2D `(azim, rad)` map via the LUT-NG pipeline. The GPU run is
/// identical to the 1D case (the 2D LUT flattens cells to `rad * bins_azim +
/// azim`, radial-major, so `bins == bins_rad * bins_azim`); only the host-side
/// packaging differs (see [`crate::common::pack_2d`]).
pub fn integrate2d_lut(
    session: &ClSession<'_>,
    inputs: &LutInputs<'_>,
    corr: &Corrections4Args,
    empty: f32,
    block_size: usize,
    bins_rad: usize,
    bins_azim: usize,
) -> Result<Result2d, String> {
    let out = run_lut_pipeline(session, inputs, corr, empty, block_size)?;
    Ok(pack_2d(out, bins_rad, bins_azim))
}
