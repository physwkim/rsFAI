//! Integrator-neutral pieces shared by the CSR and LUT OpenCL backends.
//!
//! Both `OCL_CSR_Integrator` and `OCL_LUT_Integrator` run the same NG pipeline
//! shape — per-pixel `corrections` into a `float4`, then a sparse reduction into
//! a `float8` accumulator — and expose the identical result fields. The kernels
//! and their arguments differ (see [`crate::csr`] and [`crate::lut`]); what is
//! common is the device-handle bundle, the flat→typed result packaging, and the
//! buffer-upload helper.

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_float, CL_BLOCKING};
use std::ptr;

/// The OpenCL handles a pipeline drives, always borrowed together: the `program`
/// is compiled against `context`, and `queue` is bound to it. Grouping them keeps
/// the integration entry points to a small, uniform argument list.
#[derive(Clone, Copy)]
pub struct ClSession<'a> {
    pub context: &'a Context,
    pub queue: &'a CommandQueue,
    pub program: &'a Program,
}

/// 1D integration result, one value per radial bin. `intensity`/`std`/`sem` are
/// the kernel's `averint`/`std`/`sem`; the `signal`/`variance`/`normalization`/
/// `count`/`norm_sq` columns are extracted from the `merged8` float8 accumulator
/// (`s0`/`s2`/`s4`/`s6`/`s7`; the odd lanes are Kahan low parts). This is exactly
/// pyFAI's `Integrate1dtpl` field mapping.
#[derive(Debug, Clone)]
pub struct Result1d {
    pub intensity: Vec<f32>,
    pub std: Vec<f32>,
    pub sem: Vec<f32>,
    pub signal: Vec<f32>,
    pub variance: Vec<f32>,
    pub normalization: Vec<f32>,
    pub count: Vec<f32>,
    pub norm_sq: Vec<f32>,
}

/// 2D integration result. Each field is a flat `(azim, rad)` C-order array of
/// length `bins_rad * bins_azim` — pyFAI's `field.reshape((bins_rad,
/// bins_azim)).T` layout (`integrate_ng` 2D branch). Index element `(azim, rad)`
/// as `field[azim * bins_rad + rad]`.
#[derive(Debug, Clone)]
pub struct Result2d {
    pub intensity: Vec<f32>,
    pub std: Vec<f32>,
    pub sem: Vec<f32>,
    pub signal: Vec<f32>,
    pub variance: Vec<f32>,
    pub normalization: Vec<f32>,
    pub count: Vec<f32>,
    pub norm_sq: Vec<f32>,
}

/// Flat GPU pipeline output, one value per bin. For 1D a bin is a radial bin; for
/// 2D a bin is a flattened cell (`rad * bins_azim + azim`, radial-major).
/// `merged8` is the float8 accumulator, 8 lanes per bin.
pub(crate) struct PipelineOutput {
    pub(crate) intensity: Vec<f32>,
    pub(crate) std: Vec<f32>,
    pub(crate) sem: Vec<f32>,
    pub(crate) merged8: Vec<f32>,
}

/// Allocate a read-only device buffer of `len` `f32`s, initialised from `src`
/// (zeros when `src` is `None` — the kernel skips it via its `do_*` flag).
pub(crate) fn upload_f32(
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

/// Extract `merged8` lane `c` into a per-bin vector. Lanes: 0 signal, 2 variance,
/// 4 normalization, 6 count, 7 norm_sq (odd lanes are Kahan low parts).
pub(crate) fn merged_col(merged8: &[f32], bins: usize, c: usize) -> Vec<f32> {
    (0..bins).map(|b| merged8[8 * b + c]).collect()
}

/// Reshape a flat radial-major field (`v[rad * bins_azim + azim]`) into pyFAI's
/// 2D `(azim, rad)` C-order layout (`reshape((bins_rad, bins_azim)).T`):
/// `out[azim * bins_rad + rad] = v[rad * bins_azim + azim]`.
pub(crate) fn transpose_to_azim_rad(v: &[f32], bins_rad: usize, bins_azim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; bins_rad * bins_azim];
    for rad in 0..bins_rad {
        for azim in 0..bins_azim {
            out[azim * bins_rad + rad] = v[rad * bins_azim + azim];
        }
    }
    out
}

/// Package a flat pipeline output as a 1D result (pyFAI's `Integrate1dtpl` field
/// mapping). Shared by CSR and LUT.
pub(crate) fn pack_1d(out: PipelineOutput) -> Result1d {
    let bins = out.intensity.len();
    Result1d {
        intensity: out.intensity,
        std: out.std,
        sem: out.sem,
        signal: merged_col(&out.merged8, bins, 0),
        variance: merged_col(&out.merged8, bins, 2),
        normalization: merged_col(&out.merged8, bins, 4),
        count: merged_col(&out.merged8, bins, 6),
        norm_sq: merged_col(&out.merged8, bins, 7),
    }
}

/// Package a flat pipeline output as a 2D `(azim, rad)` result (pyFAI's
/// `Integrate2dtpl`). The flat bins are radial-major cells, reshaped
/// `(bins_rad, bins_azim).T`. Shared by CSR and LUT.
pub(crate) fn pack_2d(out: PipelineOutput, bins_rad: usize, bins_azim: usize) -> Result2d {
    let bins = out.intensity.len();
    assert_eq!(
        bins,
        bins_rad * bins_azim,
        "2D bins ({bins}) != bins_rad ({bins_rad}) * bins_azim ({bins_azim})"
    );
    let t = |v: &[f32]| transpose_to_azim_rad(v, bins_rad, bins_azim);
    Result2d {
        intensity: t(&out.intensity),
        std: t(&out.std),
        sem: t(&out.sem),
        signal: t(&merged_col(&out.merged8, bins, 0)),
        variance: t(&merged_col(&out.merged8, bins, 2)),
        normalization: t(&merged_col(&out.merged8, bins, 4)),
        count: t(&merged_col(&out.merged8, bins, 6)),
        norm_sq: t(&merged_col(&out.merged8, bins, 7)),
    }
}
