# Embedded OpenCL kernels — provenance & license

These `.cl` files are **byte-identical copies of upstream pyFAI and silx OpenCL
kernels**, embedded so `rsfai-opencl` compiles the *exact* source pyFAI's
`OCL_CSR_Integrator` compiles (`src/pyFAI/opencl/azim_csr.py` `kernel_files`).
The Rust side only orchestrates these kernels; it does not modify them. The
file order in `src/program.rs` matches pyFAI's `kernel_files` list, and the
concatenation reproduces silx's `concatenate_cl_kernel` (which strips
`#include` lines — the only include is the IDE stub `for_eclipse.h`).

| File | Upstream source | License |
|---|---|---|
| `doubleword.cl` | silx `silx/resources/opencl/doubleword.cl` | MIT |
| `preprocess.cl` | pyFAI `resources/openCL/preprocess.cl` | MIT |
| `memset.cl` | pyFAI `resources/openCL/memset.cl` | MIT |
| `ocl_azim_CSR.cl` | pyFAI `resources/openCL/ocl_azim_CSR.cl` | MIT |
| `collective/reduction.cl` | pyFAI `resources/openCL/collective/reduction.cl` | MIT |
| `collective/scan.cl` | pyFAI `resources/openCL/collective/scan.cl` | MIT |
| `collective/comb_sort.cl` | pyFAI `resources/openCL/collective/comb_sort.cl` | MIT |
| `medfilt.cl` | pyFAI `resources/openCL/medfilt.cl` | MIT |

Both pyFAI (ESRF) and silx (ESRF) are MIT-licensed; the per-file copyright
headers are preserved verbatim. rsFAI is itself MIT-licensed, so redistribution
of these files is compatible. To refresh after a pyFAI/silx upgrade, re-copy
from the installed packages and re-run the `program::tests` compile check.
