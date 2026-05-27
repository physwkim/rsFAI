//! Host-side orchestration of pyFAI's CSR-NG OpenCL pipeline.
//!
//! Reproduces `OCL_CSR_Integrator.integrate_ng` (`pyFAI/opencl/azim_csr.py`):
//! `memset_ng` → `corrections4a` → `csr_integrate4`, driving pyFAI's own
//! embedded kernels (see [`crate::program`]). The Rust side only uploads
//! buffers, sets kernel arguments in pyFAI's exact order, enqueues, and reads
//! back — the arithmetic lives entirely in the GPU kernels, so on the same
//! device the result matches pyFAI's OpenCL output (validated at relative
//! error, see crate docs).
//!
//! On the Apple M4 Pro GPU `csr_integrate4`'s `CL_KERNEL_WORK_GROUP_SIZE` is
//! 256 (not 1), so pyFAI takes the work-group **tree-reduction** path
//! `csr_integrate4` (not `csr_integrate4_single`), launched with
//! `wg = wg_min = CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` (32),
//! `global = bins·wg`, `local = wg`, and `local float8 shared[wg]`. We
//! replicate that launch geometry.

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_char, cl_float, cl_int, CL_BLOCKING};
use std::ptr;

/// The 14 scalar arguments of the `corrections4a` preprocessing kernel,
/// captured verbatim from pyFAI's live `cl_kernel_args` (see the dataset's
/// `opencl_params.json`). `do_*` flags and `dtype` are `char` (i8) on the GPU;
/// `dummy`/`delta_dummy`/`normalization_factor` are `float` (f32).
#[derive(Debug, Clone, Copy)]
pub struct Corrections4aArgs {
    /// Image element type code (`_any2float`): -4 = int32, 32 = float32, …
    pub dtype: i8,
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

/// Per-pixel inputs to the pipeline. The image is the raw integer (or float)
/// buffer the GPU reinterprets via `dtype`; correction arrays are `f32` (the
/// caller casts f64→f32 exactly as pyFAI's `send_buffer` does). Absent
/// corrections are bound as zeroed buffers (their `do_*` flag is 0). The CSR
/// matrix is the exact `(data, indices, indptr)` triple the integrator was
/// built from.
pub struct CsrInputs<'a> {
    /// Raw image as i32 (the `dtype=-4` case). Length = image size.
    pub image_i32: &'a [i32],
    pub variance: Option<&'a [f32]>,
    pub dark: Option<&'a [f32]>,
    pub dark_variance: Option<&'a [f32]>,
    pub flat: Option<&'a [f32]>,
    pub solidangle: Option<&'a [f32]>,
    pub polarization: Option<&'a [f32]>,
    pub absorption: Option<&'a [f32]>,
    pub mask: Option<&'a [i8]>,
    /// CSR coefficients (f32, length nnz). For the no-split case these are all
    /// 1.0; pyFAI then binds a NULL `coefs` buffer and the kernel substitutes
    /// 1.0f. Uploading the all-ones buffer yields the identical 1.0f per entry,
    /// so the result is bit-identical and the NULL special case is unneeded.
    pub csr_data: &'a [f32],
    /// CSR column indices (i32, length nnz).
    pub csr_indices: &'a [i32],
    /// CSR row pointers (i32, length bins+1).
    pub csr_indptr: &'a [i32],
}

/// The GPU integration result, one value per radial bin. `intensity`/`std`/
/// `sem` are the kernel's `averint`/`std`/`sem`; the `signal`/`variance`/
/// `normalization`/`count`/`norm_sq` columns are extracted from the `merged8`
/// float8 accumulator (`s0`/`s2`/`s4`/`s6`/`s7`; the odd lanes are Kahan low
/// parts). This is exactly pyFAI's `Integrate1dtpl` field mapping.
#[derive(Debug, Clone)]
pub struct CsrResult1d {
    pub intensity: Vec<f32>,
    pub std: Vec<f32>,
    pub sem: Vec<f32>,
    pub signal: Vec<f32>,
    pub variance: Vec<f32>,
    pub normalization: Vec<f32>,
    pub count: Vec<f32>,
    pub norm_sq: Vec<f32>,
}

/// Allocate a read-only device buffer of `len` `f32`s, initialised from `src`
/// (zeros when `src` is `None` — the kernel skips it via its `do_*` flag).
fn upload_f32(
    context: &Context,
    queue: &CommandQueue,
    src: Option<&[f32]>,
    len: usize,
) -> Result<Buffer<cl_float>, String> {
    let host = match src {
        Some(s) => s.to_vec(),
        None => vec![0.0f32; len],
    };
    let mut buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_ONLY, len, ptr::null_mut())
            .map_err(|e| format!("alloc f32 buffer: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, &host, &[]) }
        .map_err(|e| format!("write f32 buffer: {e}"))?;
    Ok(buf)
}

/// Run pyFAI's CSR-NG OpenCL integration on `program` (which must have been
/// built by [`crate::program::build_csr_program`] for the matching bin count
/// and image size). The bin count is taken from `inputs.csr_indptr` (length
/// `bins + 1`), the single source of truth. `wg_min` is the `csr_integrate4`
/// work-group size (pyFAI's `wg_min`, 32 on the M4 Pro).
pub fn integrate1d_csr(
    context: &Context,
    queue: &CommandQueue,
    program: &Program,
    inputs: &CsrInputs<'_>,
    corr: &Corrections4aArgs,
    empty: f32,
    wg_min: usize,
) -> Result<CsrResult1d, String> {
    let size = inputs.image_i32.len();
    let nnz = inputs.csr_data.len();
    let bins = inputs.csr_indptr.len() - 1;
    assert_eq!(
        inputs.csr_indices.len(),
        nnz,
        "indices/data length mismatch"
    );

    // ---- Upload inputs ------------------------------------------------
    let mut image_buf = unsafe {
        Buffer::<cl_int>::create(context, CL_MEM_READ_ONLY, size, ptr::null_mut())
            .map_err(|e| format!("alloc image: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut image_buf, CL_BLOCKING, 0, inputs.image_i32, &[]) }
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

    let mut data_buf = unsafe {
        Buffer::<cl_float>::create(context, CL_MEM_READ_ONLY, nnz, ptr::null_mut())
            .map_err(|e| format!("alloc csr data: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut data_buf, CL_BLOCKING, 0, inputs.csr_data, &[]) }
        .map_err(|e| format!("write csr data: {e}"))?;
    let mut indices_buf = unsafe {
        Buffer::<cl_int>::create(context, CL_MEM_READ_ONLY, nnz, ptr::null_mut())
            .map_err(|e| format!("alloc csr indices: {e}"))?
    };
    unsafe {
        queue.enqueue_write_buffer(&mut indices_buf, CL_BLOCKING, 0, inputs.csr_indices, &[])
    }
    .map_err(|e| format!("write csr indices: {e}"))?;
    let mut indptr_buf = unsafe {
        Buffer::<cl_int>::create(context, CL_MEM_READ_ONLY, bins + 1, ptr::null_mut())
            .map_err(|e| format!("alloc csr indptr: {e}"))?
    };
    unsafe { queue.enqueue_write_buffer(&mut indptr_buf, CL_BLOCKING, 0, inputs.csr_indptr, &[]) }
        .map_err(|e| format!("write csr indptr: {e}"))?;

    // ---- Allocate outputs (float4 weights, float8 accumulator, results) ---
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

    // ---- memset_ng: zero averint, std, merged8 -----------------------
    // (csr_integrate4 writes every bin, so this only clears padding; we run it
    // for fidelity with integrate_ng.) Pure map: wg-independent.
    let memset = Kernel::create(program, "memset_ng").map_err(|e| format!("memset_ng: {e}"))?;
    let wg_map = 256usize;
    let g_bins = bins.div_ceil(wg_map) * wg_map;
    unsafe {
        ExecuteKernel::new(&memset)
            .set_arg(&averint_buf)
            .set_arg(&std_buf)
            .set_arg(&merged8_buf)
            .set_global_work_size(g_bins)
            .set_local_work_size(wg_map)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue memset_ng: {e}"))?;
    }

    // ---- corrections4a: per-pixel preprocessing into output4 ---------
    let corrections =
        Kernel::create(program, "corrections4a").map_err(|e| format!("corrections4a: {e}"))?;
    let g_data = size.div_ceil(wg_map) * wg_map;
    unsafe {
        ExecuteKernel::new(&corrections)
            .set_arg(&image_buf)
            .set_arg(&(corr.dtype as cl_char))
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
            .set_local_work_size(wg_map)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue corrections4a: {e}"))?;
    }

    // ---- csr_integrate4: tree-reduction over CSR rows ----------------
    // wg = wg_min, one work-group per bin; shared = wg_min · sizeof(float8).
    let integrate =
        Kernel::create(program, "csr_integrate4").map_err(|e| format!("csr_integrate4: {e}"))?;
    let shared_bytes = wg_min * std::mem::size_of::<[cl_float; 8]>();
    unsafe {
        ExecuteKernel::new(&integrate)
            .set_arg(&output4_buf)
            .set_arg(&data_buf)
            .set_arg(&indices_buf)
            .set_arg(&indptr_buf)
            .set_arg(&(bins as cl_int))
            .set_arg(&(empty as cl_float))
            .set_arg(&(corr.error_model as cl_char))
            .set_arg(&merged8_buf)
            .set_arg(&averint_buf)
            .set_arg(&std_buf)
            .set_arg(&sem_buf)
            .set_arg_local_buffer(shared_bytes)
            .set_global_work_size(bins * wg_min)
            .set_local_work_size(wg_min)
            .enqueue_nd_range(queue)
            .map_err(|e| format!("enqueue csr_integrate4: {e}"))?;
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

    // merged8 lanes: s0 signal, s2 variance, s4 normalization, s6 count,
    // s7 norm_sq (odd lanes are Kahan low parts, not exposed).
    let col = |c: usize| (0..bins).map(|b| merged8[8 * b + c]).collect::<Vec<f32>>();
    Ok(CsrResult1d {
        intensity,
        std,
        sem,
        signal: col(0),
        variance: col(2),
        normalization: col(4),
        count: col(6),
        norm_sq: col(7),
    })
}
