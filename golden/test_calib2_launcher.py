#!/usr/bin/env python3
"""Validate the ``rsfai_calib2`` launcher hook for the pyFAI calibration GUI.

Run in the ``daq`` conda env:

    OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python golden/test_calib2_launcher.py

The GUI's ``IntegrationTask.run`` constructs ``AzimuthalIntegrator(dist=, poni1=,
..., detector=, wavelength=)``, builds the method via ``method_registry`` and
calls ``integrate1d_ng`` / ``integrate2d`` with ``data/npt/unit/mask/
polarization_factor/dark/flat``.  The launcher rebinds that constructed class to
``rsfai_backend.RsfaiAzimuthalIntegrator``.  This test reproduces that exact
construction + call sequence (with a runtime mask + dark + flat, which the golden
never exercises) on the rsFAI subclass and asserts:

  * a CPU (cython) method runs on rsFAI (``compute_engine`` starts with
    ``rsfai``) and is BIT-EXACT vs a plain pyFAI integrator built identically;
  * an OpenCL csr method on the integer image takes the rsFAI ``GpuEngine`` path
    and matches a plain pyFAI OpenCL run (CSR is deterministic on this device).

It also runs the launcher's ``install_backend()`` + ``brand_gui()`` against the
real GUI module when a Qt binding is present: it asserts the integrator rebind
took, then constructs a ``CalibrationWindow`` offscreen and asserts the rebranded
window title ("rsFAI Calibration", vs the .ui default "PyFAI Calibration") and
the Qt application display name.  Without a Qt binding that leg SKIPs and is
reported as unverified rather than silently passing.
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pyFAI  # noqa: E402
from pyFAI import method_registry  # noqa: E402
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsfai_backend  # noqa: E402
import rsfai_calib2  # noqa: E402

# A Pilatus1M dataset (geometry.poni + an int32 image) — geometry only; this test
# does not read its golden, it compares rsFAI vs a live pyFAI run side by side.
DS = HERE / "datasets" / "Pilatus1M__bbox-csr-opencl__q_nmm1__npt1000__errpoisson"
ACC = ("intensity", "sigma", "count", "sum_signal", "sum_variance",
       "sum_normalization", "sum_normalization2", "std", "sem", "radial", "azimuthal")


def geometry_params():
    geo = pyFAI.load(str(DS / "geometry.poni"))
    return dict(dist=geo.dist, poni1=geo.poni1, poni2=geo.poni2,
                rot1=geo.rot1, rot2=geo.rot2, rot3=geo.rot3,
                detector=geo.detector, wavelength=geo.wavelength)


def select(method_tuple):
    """Reproduce IntegrationTask.run's method selection for dim 1 and 2."""
    split, algo, impl = method_tuple
    base = method_registry.Method(0, split, algo, impl, None)
    out = []
    for dim in (1, 2):
        m = base.fixed(dim=dim)
        found = method_registry.IntegrationMethod.select_method(method=m)
        out.append(found[0].method if found else
                   method_registry.Method(dim, m.split, "*", "*", None))
    return out


def gui_run(cls, params, image, mask, dark, flat, method_tuple):
    """The construction + integrate calls IntegrationTask.run performs."""
    method1d, method2d = select(method_tuple)
    ai = cls(**params)
    ai.enable_parallax(False)
    common = dict(unit="q_nm^-1", mask=mask, polarization_factor=0.99, dark=dark, flat=flat)
    r1 = ai.integrate1d_ng(method=method1d, data=image, npt=1000, **common)
    r2 = ai.integrate2d(method=method2d, data=image, npt_rad=1000, npt_azim=360, **common)
    return r1, r2


def fields(res):
    return {a: getattr(res, a) for a in ACC
            if isinstance(getattr(res, a, None), np.ndarray)}


def cmp(a_fields, b_fields, exact, tol, tag, log):
    ok_all = True
    for k in a_fields:
        if k not in b_fields:
            continue
        a = np.ascontiguousarray(a_fields[k]).astype(np.float64).ravel()
        b = np.ascontiguousarray(b_fields[k]).astype(np.float64).ravel()
        if a.shape != b.shape:
            log.append(f"    {tag}.{k:18s} FAIL shape {a.shape}!={b.shape}")
            ok_all = False
            continue
        if exact:
            ok = a_fields[k].tobytes() == b_fields[k].tobytes()
            detail = "bit-exact" if ok else "NOT bit-exact"
        else:
            denom = np.maximum(np.abs(b), 1e-30)
            e = float(np.max(np.abs(a - b) / denom))
            ok = e <= tol
            detail = f"rel={e:.2e}"
        ok_all = ok_all and ok
        log.append(f"    {tag}.{k:18s} {'PASS' if ok else 'FAIL'} | {detail}")
    return ok_all


def main():
    print(f"pyFAI {pyFAI.version} | rsfai_calib2 launcher hook | "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}\n")
    params = geometry_params()
    image = np.ascontiguousarray(np.load(DS / "image.npy"))   # int32
    shape = image.shape
    # A runtime user mask + dark + flat — the paths the golden (mask=None, no
    # dark/flat) never covers, exactly what the GUI may pass.
    mask = np.zeros(shape, dtype=np.int8)
    mask[100:140, 200:260] = 1
    dark = np.full(shape, 1.5, dtype=np.float32)
    flat = np.full(shape, 1.05, dtype=np.float32)

    Rs = rsfai_backend.RsfaiAzimuthalIntegrator
    failures = 0

    # --- CPU (cython): rsFAI path, bit-exact vs plain pyFAI ---------------
    print("=== GUI run, method=(bbox,csr,cython) ===")
    r1, r2 = gui_run(Rs, params, image, mask, dark, flat, ("bbox", "csr", "cython"))
    p1, p2 = gui_run(AzimuthalIntegrator, params, image, mask, dark, flat, ("bbox", "csr", "cython"))
    log = []
    eng1, eng2 = str(r1.compute_engine), str(r2.compute_engine)
    print(f"  1d engine: {eng1}\n  2d engine: {eng2}")
    if not (eng1.startswith("rsfai") and eng2.startswith("rsfai")):
        print("  routing            FAIL | expected rsFAI on both"); failures += 1
    else:
        print("  routing            PASS | rsFAI on both")
    ok = cmp(fields(r1), fields(p1), True, 0.0, "1d", log)
    ok &= cmp(fields(r2), fields(p2), True, 0.0, "2d", log)
    print("\n".join(log))
    failures += (not ok)
    print()

    # --- OpenCL csr on the integer image: GpuEngine path ------------------
    print("=== GUI run, method=(bbox,csr,opencl) on the integer image ===")
    r1, r2 = gui_run(Rs, params, image, mask, dark, flat, ("bbox", "csr", "opencl"))
    p1, p2 = gui_run(AzimuthalIntegrator, params, image, mask, dark, flat, ("bbox", "csr", "opencl"))
    log = []
    eng1, eng2 = str(r1.compute_engine), str(r2.compute_engine)
    print(f"  1d engine: {eng1}\n  2d engine: {eng2}")
    if not (eng1.startswith("rsfai") and eng2.startswith("rsfai")):
        print("  routing            FAIL | expected rsFAI GPU on both"); failures += 1
    else:
        print("  routing            PASS | rsFAI GPU on both")
    ok = cmp(fields(r1), fields(p1), False, 1e-6, "1d", log)
    ok &= cmp(fields(r2), fields(p2), False, 1e-6, "2d", log)
    print("\n".join(log))
    failures += (not ok)
    print()

    # --- The launcher's install_backend() + brand_gui() against the real GUI -
    print("=== launcher install_backend() + brand_gui() ===")
    try:
        patched = rsfai_calib2.install_backend()
    except ImportError as exc:
        print(f"  SKIP (unverified): GUI import needs a Qt binding — {exc}")
    else:
        from pyFAI.gui.tasks import IntegrationTask
        ok = IntegrationTask.AzimuthalIntegrator is Rs and bool(patched)
        print(f"  patched {patched}; IntegrationTask.AzimuthalIntegrator is rsFAI subclass: "
              f"{IntegrationTask.AzimuthalIntegrator is Rs}")
        if not ok:
            failures += 1

        # Rebrand: build a CalibrationWindow offscreen and assert the title now
        # reads as rsFAI (the .ui default is "PyFAI Calibration").
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from silx.gui import qt
        from pyFAI.gui.CalibrationWindow import CalibrationWindow
        from pyFAI.gui.CalibrationContext import CalibrationContext

        app = qt.QApplication.instance() or qt.QApplication([])  # noqa: F841
        rsfai_calib2.brand_gui()
        settings = qt.QSettings(qt.QSettings.IniFormat, qt.QSettings.UserScope,
                                "pyfai", "pyfai-calib2-rsfai-test", None)
        context = CalibrationContext(settings)
        context.restoreSettings()
        window = CalibrationWindow(context)
        title = window.windowTitle()
        appname = qt.QApplication.instance().applicationDisplayName()
        title_ok = (title == rsfai_calib2._WINDOW_TITLE
                    and appname == rsfai_calib2._APP_DISPLAY_NAME)
        print(f"  window title: {title!r}; appDisplayName: {appname!r} | "
              f"{'PASS' if title_ok else 'FAIL'}")
        if not title_ok:
            failures += 1

    print(f"\nRESULT: {'PASS' if failures == 0 else 'FAIL'} ({failures} failing groups)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
