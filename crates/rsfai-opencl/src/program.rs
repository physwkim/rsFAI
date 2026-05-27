//! Assembly and compilation of pyFAI's OpenCL CSR and LUT programs.
//!
//! pyFAI's `OCL_CSR_Integrator` / `OCL_LUT_Integrator` each compile a single
//! program from a fixed list of `.cl` files (their `kernel_files`), concatenated
//! in order by silx's `concatenate_cl_kernel`, which strips `#include` lines
//! (silx `read_cl_file`). We embed byte-identical copies of those files (silx's
//! `doubleword.cl` + pyFAI's kernels, both MIT-licensed) and reproduce the same
//! concatenation and the same compile flags, so the GPU compiles the identical
//! source pyFAI does.

use opencl3::context::Context;
use opencl3::program::Program;

/// CSR kernel sources, in `azim_csr.py`'s `kernel_files` order. `doubleword.cl`
/// is silx's; the rest are pyFAI's `resources/openCL/`. Concatenation order
/// matters: later files (e.g. `ocl_azim_CSR.cl`) call functions defined in
/// earlier ones (`doubleword.cl`).
const CSR_KERNEL_SOURCES: &[&str] = &[
    include_str!("../kernels/doubleword.cl"),
    include_str!("../kernels/preprocess.cl"),
    include_str!("../kernels/memset.cl"),
    include_str!("../kernels/ocl_azim_CSR.cl"),
    include_str!("../kernels/collective/reduction.cl"),
    include_str!("../kernels/collective/scan.cl"),
    include_str!("../kernels/collective/comb_sort.cl"),
    include_str!("../kernels/medfilt.cl"),
];

/// LUT kernel sources, in `azim_lut.py`'s `kernel_files` order: `doubleword.cl`,
/// `preprocess.cl`, `memset.cl`, `ocl_azim_LUT.cl`.
const LUT_KERNEL_SOURCES: &[&str] = &[
    include_str!("../kernels/doubleword.cl"),
    include_str!("../kernels/preprocess.cl"),
    include_str!("../kernels/memset.cl"),
    include_str!("../kernels/ocl_azim_LUT.cl"),
];

/// Concatenate kernel sources the way silx's `concatenate_cl_kernel` does: join
/// the files, dropping every line that starts with `#include ` (silx's "dummy
/// preprocessor" — the only includes are the IDE stub `for_eclipse.h`).
fn concatenate(sources: &[&str]) -> String {
    let mut out = String::new();
    for src in sources {
        for line in src.lines() {
            if line.starts_with("#include ") {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// The concatenated CSR program source.
pub fn csr_program_source() -> String {
    concatenate(CSR_KERNEL_SOURCES)
}

/// The concatenated LUT program source.
pub fn lut_program_source() -> String {
    concatenate(LUT_KERNEL_SOURCES)
}

/// Compile options matching pyFAI/silx on the Apple M4 Pro:
/// `-Dcl_khr_fp64=0` (silx `apple_gpu_option`: the Apple driver wrongly claims
/// fp64, so silx forces the kernels onto the doubleword path; `x87_volatile` is
/// empty on this device) plus the per-integrator `-D NBINS -D NIMAGE`.
pub fn csr_compile_options(nbins: usize, nimage: usize) -> String {
    format!("-Dcl_khr_fp64=0 -D NBINS={nbins} -D NIMAGE={nimage}")
}

/// LUT compile options: the CSR options plus `-D NLUT` (densified LUT width) and
/// `-D ON_CPU=0` (GPU device — selects the transposed `lut[j*NBINS+bin]` access
/// path in `ocl_azim_LUT.cl`).
pub fn lut_compile_options(nbins: usize, nimage: usize, nlut: usize) -> String {
    format!("-Dcl_khr_fp64=0 -D NBINS={nbins} -D NIMAGE={nimage} -D NLUT={nlut} -D ON_CPU=0")
}

/// Build the CSR program for a given bin count and image size on `context`. On
/// failure the `Err` is the OpenCL build log (what
/// `create_and_build_from_source` returns), so a compile error is legible.
pub fn build_csr_program(
    context: &Context,
    nbins: usize,
    nimage: usize,
) -> Result<Program, String> {
    let source = csr_program_source();
    let options = csr_compile_options(nbins, nimage);
    Program::create_and_build_from_source(context, &source, &options)
}

/// Build the LUT program for a given bin count, image size and densified LUT
/// width on `context`. `Err` carries the OpenCL build log on failure.
pub fn build_lut_program(
    context: &Context,
    nbins: usize,
    nimage: usize,
    nlut: usize,
) -> Result<Program, String> {
    let source = lut_program_source();
    let options = lut_compile_options(nbins, nimage, nlut);
    Program::create_and_build_from_source(context, &source, &options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{create_queue, default_context};
    use opencl3::kernel::Kernel;

    #[test]
    fn pyfai_csr_kernels_compile_on_device() {
        // The #include stub must be gone, and the doubleword primitives present.
        let src = csr_program_source();
        assert!(!src.contains("#include "), "#include not stripped");
        assert!(src.contains("dw_plus_dw"), "doubleword.cl not concatenated");
        assert!(
            src.contains("csr_integrate4_single"),
            "ocl_azim_CSR.cl not concatenated"
        );

        let (context, _device) = default_context(true).expect("context");
        let _queue = create_queue(&context).expect("queue");
        // Representative sizes (npt=1000, Pilatus1M pixel count).
        let program = build_csr_program(&context, 1000, 981 * 1043).expect("CSR program build");

        // Every kernel the CSR-NG path enqueues must resolve.
        for name in [
            "memset_ng",
            "corrections4a",
            "csr_integrate4",
            "csr_integrate4_single",
        ] {
            Kernel::create(&program, name).unwrap_or_else(|e| panic!("kernel {name}: {e}"));
        }
    }

    #[test]
    fn pyfai_lut_kernels_compile_on_device() {
        let src = lut_program_source();
        assert!(!src.contains("#include "), "#include not stripped");
        assert!(src.contains("dw_plus_dw"), "doubleword.cl not concatenated");
        assert!(
            src.contains("lut_integrate4"),
            "ocl_azim_LUT.cl not concatenated"
        );

        let (context, _device) = default_context(true).expect("context");
        let _queue = create_queue(&context).expect("queue");
        // Representative sizes (npt=1000, Pilatus1M pixel count, lut_size=1792).
        let program =
            build_lut_program(&context, 1000, 981 * 1043, 1792).expect("LUT program build");

        // Every kernel the LUT-NG path enqueues must resolve.
        for name in ["memset_out", "corrections4", "lut_integrate4"] {
            Kernel::create(&program, name).unwrap_or_else(|e| panic!("kernel {name}: {e}"));
        }
    }
}
