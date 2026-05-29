#!/usr/bin/env python3
"""Launch the pyFAI calibration GUI (``pyFAI-calib2``) with rsFAI as the
integration backend.

The calibration GUI's integration panel (``pyFAI.gui.tasks.IntegrationTask``)
constructs a fresh ``AzimuthalIntegrator`` per run from the module-level symbol
it imported, then drives it with ``integrate1d_ng`` / ``integrate2d``.  This
launcher rebinds that module symbol to :class:`rsfai_backend.RsfaiAzimuthalIntegrator`
*before* handing off to ``pyFAI.app.calib2.main``, so every integration in the
running GUI executes on rsFAI's Rust kernels (CPU) or its ``GpuEngine`` (OpenCL
csr/lut on an integer image); any configuration rsFAI does not cover falls back
to pyFAI transparently and produces the identical result.

Rebinding the *source* module (``pyFAI.integrator.azimuthal``) would be too late:
``IntegrationTask`` already imported the name into its own namespace, so the
patch is applied there (and to any other GUI module that constructs one).

Usage — identical arguments to ``pyFAI-calib2``::

    python golden/rsfai_calib2.py [pyFAI-calib2 args...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rsfai_backend

# Every GUI module that binds the ``AzimuthalIntegrator`` name in its own
# namespace and constructs one for integration. Rebound to the rsFAI subclass.
_PATCH_TARGETS = ("pyFAI.gui.tasks.IntegrationTask",)


def install_backend():
    """Rebind ``AzimuthalIntegrator`` to the rsFAI subclass in each GUI module
    that constructs one.  Returns the list of modules actually patched."""
    import importlib

    patched = []
    for modname in _PATCH_TARGETS:
        module = importlib.import_module(modname)
        if getattr(module, "AzimuthalIntegrator", None) is not None:
            module.AzimuthalIntegrator = rsfai_backend.RsfaiAzimuthalIntegrator
            patched.append(modname)
    return patched


def main():
    patched = install_backend()
    print(f"rsFAI backend active in: {', '.join(patched) or '(none)'}", file=sys.stderr)
    from pyFAI.app.calib2 import main as calib2_main
    return calib2_main()


if __name__ == "__main__":
    sys.exit(main())
