//! OpenCL/GPU backend for rsFAI (Phase 2).
//!
//! The cython-parity crates (`rsfai-integrate` et al.) reproduce pyFAI's CPU
//! kernels **bit-exactly**. This crate targets pyFAI's **OpenCL** engines
//! instead, which run on the GPU. On the target device (Apple M4 Pro) there is
//! no f64 support, so pyFAI accumulates in a *doubleword* (two-f32 Kahan)
//! representation; the GPU's work-group reductions also reorder additions. So
//! this backend is **tolerance-validated, not bit-exact** against pyFAI's
//! OpenCL output.
//!
//! Fidelity strategy: rather than rewrite the kernels in a second language (and
//! risk silent arithmetic drift), this backend **reuses pyFAI's own
//! MIT-licensed `.cl` kernel source** — the same files pyFAI compiles, in the
//! same concatenation order, with the same `-D` flags — and ports only the
//! host-side orchestration (buffer upload, kernel enqueue, read-back) to Rust
//! via `opencl3`. Same kernel + same device + same compiler keeps the GPU
//! arithmetic identical to pyFAI's; the residual tolerance comes only from the
//! non-deterministic cross-work-group reduction order.
//!
//! This module currently provides device discovery and a toolchain smoke test;
//! the CSR/LUT integration orchestrators are added incrementally.

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL};
use opencl3::error_codes::ClError;

pub mod csr;
pub mod program;

pub use csr::{
    integrate1d_csr, integrate2d_csr, ClSession, Corrections4aArgs, CsrInputs, CsrResult1d,
    CsrResult2d,
};

/// A discovered OpenCL device, summarised for backend selection.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// OpenCL device id.
    pub id: *mut std::ffi::c_void,
    /// `CL_DEVICE_NAME`, e.g. "Apple M4 Pro".
    pub name: String,
    /// `CL_DEVICE_VENDOR`.
    pub vendor: String,
    /// `CL_DEVICE_TYPE` bitfield (4 = GPU).
    pub dev_type: u64,
    /// `CL_DEVICE_DOUBLE_FP_CONFIG`; 0 means the device has no f64 support and
    /// pyFAI's doubleword kernels are required.
    pub double_fp_config: u64,
    /// Maximum work-group size the device reports.
    pub max_work_group_size: usize,
}

/// Enumerate every OpenCL device visible on this machine.
pub fn list_devices() -> Result<Vec<DeviceInfo>, ClError> {
    let mut out = Vec::new();
    for id in get_all_devices(CL_DEVICE_TYPE_ALL)? {
        let d = Device::new(id);
        out.push(DeviceInfo {
            id,
            name: d.name()?,
            vendor: d.vendor()?,
            dev_type: d.dev_type()?,
            double_fp_config: d.double_fp_config().unwrap_or(0),
            max_work_group_size: d.max_work_group_size()?,
        });
    }
    Ok(out)
}

/// Build a `Context` on the first device matching `prefer_gpu` (a GPU if one
/// exists, else the first device). Returns the context and the chosen device.
pub fn default_context(prefer_gpu: bool) -> Result<(Context, Device), ClError> {
    let ids = get_all_devices(CL_DEVICE_TYPE_ALL)?;
    let id = if prefer_gpu {
        ids.iter()
            .copied()
            .find(|&i| Device::new(i).dev_type().unwrap_or(0) == 4)
            .or_else(|| ids.first().copied())
    } else {
        ids.first().copied()
    }
    .ok_or(ClError(opencl3::error_codes::CL_DEVICE_NOT_FOUND))?;
    let device = Device::new(id);
    let context = Context::from_device(&device)?;
    Ok((context, device))
}

/// Create a command queue on the context's default device.
///
/// We deliberately call the OpenCL 1.x `clCreateCommandQueue`
/// (opencl3's `create_default`), not the 2.0
/// `clCreateCommandQueueWithProperties`. The target Apple platform is OpenCL
/// 1.2 only: cl3 loads functions dynamically and the 2.0 entry point is absent,
/// so the 2.0 path fails with `DLOPEN_FUNCTION_NOT_AVAILABLE` (-2001). The
/// `#[deprecated]` warning fires only because opencl3 is compiled with the 2.0+
/// feature set; for a 1.2 device the 1.x call is the correct and only-supported
/// one (pyopencl/silx use it on Apple too). Localised here so the rest of the
/// backend stays warning-clean.
#[allow(deprecated)]
pub fn create_queue(context: &Context) -> Result<CommandQueue, ClError> {
    CommandQueue::create_default(context, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    use opencl3::program::Program;
    use opencl3::types::{cl_float, CL_NON_BLOCKING};
    use std::ptr;

    #[test]
    fn enumerates_a_device() {
        let devs = list_devices().expect("device enumeration");
        assert!(!devs.is_empty(), "no OpenCL device found");
        for d in &devs {
            eprintln!(
                "device: {} ({}) | type={} | double_fp_config={} | max_wg={}",
                d.name, d.vendor, d.dev_type, d.double_fp_config, d.max_work_group_size
            );
        }
    }

    /// End-to-end toolchain check: compile a trivial kernel from source, run it
    /// on the device, and read back the result. This de-risks the whole backend
    /// (opencl3 links the Apple OpenCL framework and a kernel actually executes)
    /// before any pyFAI-kernel orchestration is built on top.
    #[test]
    fn compiles_and_runs_a_kernel() {
        const SRC: &str = r#"
            __kernel void vadd(__global const float* a,
                               __global const float* b,
                               __global float* c) {
                size_t i = get_global_id(0);
                c[i] = a[i] + b[i];
            }
        "#;
        let (context, _device) = default_context(true).expect("context");
        let queue = create_queue(&context).expect("command queue");
        let program =
            Program::create_and_build_from_source(&context, SRC, "").expect("program build");
        let kernel = Kernel::create(&program, "vadd").expect("kernel");

        let n = 1024usize;
        let a: Vec<cl_float> = (0..n).map(|i| i as f32).collect();
        let b: Vec<cl_float> = (0..n).map(|i| (2 * i) as f32).collect();

        let mut a_buf = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .expect("a buffer")
        };
        let mut b_buf = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .expect("b buffer")
        };
        let c_buf = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, n, ptr::null_mut())
                .expect("c buffer")
        };

        unsafe {
            queue
                .enqueue_write_buffer(&mut a_buf, CL_NON_BLOCKING, 0, &a, &[])
                .expect("write a");
            queue
                .enqueue_write_buffer(&mut b_buf, CL_NON_BLOCKING, 0, &b, &[])
                .expect("write b");
        }

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&a_buf)
                .set_arg(&b_buf)
                .set_arg(&c_buf)
                .set_global_work_size(n)
                .enqueue_nd_range(&queue)
                .expect("enqueue")
        };
        kernel_event.wait().expect("kernel wait");

        let mut c = vec![0.0f32; n];
        unsafe {
            queue
                .enqueue_read_buffer(&c_buf, CL_NON_BLOCKING, 0, &mut c, &[])
                .expect("read c")
        }
        .wait()
        .expect("read wait");

        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i], "mismatch at {i}");
        }
    }
}
