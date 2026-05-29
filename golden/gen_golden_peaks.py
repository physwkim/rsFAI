#!/usr/bin/env python
"""Golden generator for the peak-finding primitives (rsfai-peaks).

Run single-thread in the daq env (pyFAI 2026.5.0, built -ffp-contract=off):

    env -u VIRTUAL_ENV CONDA_PREFIX=/Users/stevek/mamba/envs/daq \\
        OMP_NUM_THREADS=1 /Users/stevek/mamba/envs/daq/bin/python \\
        golden/gen_golden_peaks.py

Five parity surfaces, all feeding crates/rsfai-peaks/tests/golden_peaks.rs:

  * label: scipy.ndimage.label on a synthetic binary image, 8- and
    4-connectivity (the get_labeled_massif connected-component pass,
    massif.py:365). Output is the int32 label image + the component count.

  * edt: scipy.ndimage.distance_transform_edt(mask, return_indices=True) on a
    synthetic boolean mask (the Massif cleaned_data in-fill, massif.py:295).
    Output is the f64 distance image and the two int32 feature-index arrays.

  * watershed: pyFAI.ext.watershed.InverseWatershed on a deterministic
    multi-Gaussian image; dump the int32 labels, the uint8 borders, and the
    peaks_from_area coordinates for a few (Imin, keep, refine, dmin) configs.

  * blob: pyFAI.ext._blob.local_max on a DoG stack produced by a real
    BlobDetection octave, and the refine_Hessian sub-pixel refinement of each
    detected keypoint. The DoG stack (float32) is dumped as the Rust input, so
    the gaussian smoothing (a Cython/scipy black box) is out of the comparison;
    the deterministic detection + refinement is what is gated.

  * ellipse: pyFAI.utils.ellipse.fit_ellipse on the test_utils_ellipse fixtures.
    Dump the points, the numpy design matrix D (bit-exact building block), and
    the fitted ellipse parameters (Tier-B, eigensolver tolerance).

Everything here is small (synthetic images <= 128x128, a handful of points), so
the whole tree is committed: the .npy arrays and the manifest. Provenance
(pyFAI/numpy/scipy version, per-field dtype) goes in manifest.json.
"""

import json
import os

import numpy as np
import scipy

import pyFAI
from pyFAI.ext import watershed
from pyFAI.ext import _blob
from pyFAI.blob_detection import BlobDetection
from pyFAI.utils import ellipse as ellipse_mdl
from scipy.ndimage import label as scipy_label
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "datasets_peaks")


def save(name, arr):
    arr = np.ascontiguousarray(arr)
    np.save(os.path.join(OUTDIR, name), arr)
    return {"file": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def gaussian_image(shape, peaks, dtype=np.float32):
    """Deterministic image: sum of isotropic Gaussians at known centres."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    img = np.zeros(shape, np.float64)
    for (yc, xc, sigma, amp) in peaks:
        img += amp * np.exp(-((yy - yc) ** 2 + (xx - xc) ** 2) / (2.0 * sigma * sigma))
    return img.astype(dtype)


def ring_image(shape, y0, x0, sigma, n_rings, mod):
    """Deterministic concentric-ring image: a modulated radial Gaussian comb.
    This produces real difference-of-Gaussian scale-space extrema (isolated
    Gaussian blobs do not, within a single octave), so the blob detector has
    keypoints to find."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    r = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2).astype(np.float64)
    chi = np.arctan2(yy - y0, xx - x0)
    img = np.zeros(shape, np.float64)
    for rad in np.linspace(5.0, r.max() * 0.9, n_rings):
        img += np.exp(-((r - rad) ** 2) / (2.0 * sigma * sigma))
    return img * (1.0 + np.sin(0.5 * r + chi * mod))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = {
        "dataset": "peaks",
        "pyfai_version": pyFAI.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": os.uname().sysname + "-" + os.uname().machine,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }

    # ---- label: scipy.ndimage.label, 8- and 4-connectivity ----------------
    # Synthetic binary image with several disjoint and bridging blobs so the
    # 8- vs 4-connectivity difference and the equivalence resolution are both
    # exercised.
    lab_in = np.zeros((24, 28), np.int8)
    lab_in[2:5, 3:7] = 1
    lab_in[2:5, 9:12] = 1
    lab_in[6:9, 5:6] = 1  # diagonal bridge between the two top blobs (8-conn only)
    lab_in[5, 6] = 1
    lab_in[5, 8] = 1
    lab_in[12:16, 4:8] = 1
    lab_in[12:16, 20:24] = 1
    lab_in[18:22, 10:14] = 1
    label_meta = []
    label_meta.append(save("label_input.npy", lab_in.astype(np.int8)))
    label_cases = []
    for conn, struct in [
        ("c8", np.ones((3, 3), np.int8)),
        ("c4", np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.int8)),
    ]:
        lab, n = scipy_label(lab_in > 0, struct)
        m = save(f"label_{conn}.npy", lab.astype(np.int32))
        m["connectivity"] = conn
        m["n"] = int(n)
        label_cases.append(m)
    manifest["label"] = {"input": label_meta[0], "cases": label_cases}

    # ---- edt: distance_transform_edt(mask, return_indices=True) -----------
    # A boolean mask (True == foreground, distance to nearest False is computed),
    # mirroring how Massif.cleaned_data calls EDT on the invalid-pixel mask.
    rng = np.random.default_rng(12345)
    edt_in = (rng.random((20, 22)) < 0.55)
    # carve a couple of guaranteed background pixels so every region is finite
    edt_in[0, 0] = False
    edt_in[-1, -1] = False
    dist, idx = distance_transform_edt(edt_in, return_distances=True, return_indices=True)
    edt_meta = {
        "input": save("edt_input.npy", edt_in.astype(np.int8)),
        "dist": save("edt_dist.npy", dist.astype(np.float64)),
        "idx_row": save("edt_idx_row.npy", idx[0].astype(np.int32)),
        "idx_col": save("edt_idx_col.npy", idx[1].astype(np.int32)),
    }
    manifest["edt"] = edt_meta

    # ---- watershed: InverseWatershed on a deterministic image -------------
    ws_img = gaussian_image(
        (64, 64),
        [
            (16, 16, 4.0, 100.0),
            (16, 48, 4.0, 80.0),
            (48, 16, 4.0, 90.0),
            (48, 48, 4.0, 120.0),
            (32, 32, 3.0, 60.0),
        ],
    )
    w = watershed.InverseWatershed(data=ws_img)
    w.init()
    ws_meta = {
        "image": save("ws_image.npy", ws_img.astype(np.float32)),
        "labels": save("ws_labels.npy", np.ascontiguousarray(w.labels).astype(np.int32)),
        "borders": save("ws_borders.npy", np.ascontiguousarray(w.borders).astype(np.int32)),
        "n_regions": len(set(w.regions.values())),
    }
    mask_all = np.ones(ws_img.shape, np.uint8)
    ws_cases = []
    configs = [
        ("imin10_keep10_refine", ws_img, "ws_image.npy", dict(Imin=10.0, keep=10, refine=True, dmin=0.0)),
        ("imin10_keep10_norefine", ws_img, "ws_image.npy", dict(Imin=10.0, keep=10, refine=False, dmin=0.0)),
        ("imin0_keep3_refine_dmin5", ws_img, "ws_image.npy", dict(Imin=1.0, keep=3, refine=True, dmin=5.0)),
    ]
    # Dense grid of equal-ish-intensity peaks: exercises the many-region path
    # and tied intensities, where the coordinate SET must still match pyFAI
    # even though the returned list order follows CPython set-hash iteration.
    dense_img = gaussian_image(
        (96, 96),
        [
            (yc, xc, 3.0, 50.0 + float((yc * 7 + xc * 3) % 80))
            for yc in range(10, 90, 16)
            for xc in range(10, 90, 16)
        ],
    )
    save("ws_dense_image.npy", dense_img.astype(np.float32))
    configs.append(
        ("dense_imin5_keep100_refine", dense_img, "ws_dense_image.npy",
         dict(Imin=5.0, keep=100, refine=True, dmin=0.0))
    )
    for tag, img_for_case, img_file, kw in configs:
        # fresh object per case (peaks_from_area is read-only, but be safe)
        wi = watershed.InverseWatershed(data=img_for_case)
        wi.init()
        mask_for_case = np.ones(img_for_case.shape, np.uint8)
        pts = wi.peaks_from_area(mask_for_case, **kw)
        arr = np.array(pts, dtype=np.float64).reshape(-1, 2) if pts else np.zeros((0, 2), np.float64)
        m = save(f"ws_peaks_{tag}.npy", arr)
        m["tag"] = tag
        m["image"] = img_file
        m["shape"] = list(img_for_case.shape)
        m["config"] = {k: (float(v) if isinstance(v, float) else v) for k, v in kw.items()}
        ws_cases.append(m)
    ws_meta["peaks"] = ws_cases
    manifest["watershed"] = ws_meta

    # ---- blob: DoG local_max + refine_Hessian -----------------------------
    # Build a real DoG octave from a deterministic ring image, dump the DoG
    # stack (float32) + the detection / refinement results.
    blob_img = ring_image((128, 128), 64.0, 64.0, 2.0, 10, 8)
    bd = BlobDetection(blob_img)
    bd._one_octave(shrink=False, refine=False, n_5=False)
    dogs = np.ascontiguousarray(bd.dogs, dtype=np.float32)
    cur_mask = np.ascontiguousarray(bd.cur_mask, dtype=np.int8)
    blob_meta = {
        "dogs": save("blob_dogs.npy", dogs),
        "mask": save("blob_mask.npy", cur_mask),
    }
    blob_cases = []
    for n_5 in (False, True):
        is_max = _blob.local_max(dogs, cur_mask, n_5)
        kps, kpy, kpx = np.where(is_max)
        coords = np.stack([kps, kpy, kpx], axis=1).astype(np.int32) if kps.size else np.zeros((0, 3), np.int32)
        tag = "n5" if n_5 else "n3"
        cm = save(f"blob_localmax_{tag}.npy", coords)
        cm["tag"] = tag
        cm["n_5"] = bool(n_5)
        blob_cases.append(cm)
        # refine_Hessian for the n_5=False keypoints (the process() default uses
        # n_5=True; we refine whatever was detected for this case).
        if n_5 is False and kps.size:
            rx, ry, rs, peakval, valid = bd.refine_Hessian(kpx, kpy, kps)
            ref = np.stack(
                [rx.astype(np.float64), ry.astype(np.float64), rs.astype(np.float64),
                 peakval.astype(np.float64), valid.astype(np.float64)],
                axis=1,
            )
            rm = save("blob_refine_n3.npy", ref)
            rm["tag"] = "n3"
            blob_meta["refine"] = rm
    blob_meta["localmax"] = blob_cases
    manifest["blob"] = blob_meta

    # ---- ellipse: fit_ellipse on the test fixtures ------------------------
    ellipse_cases = []
    fixtures = []
    angles = np.arange(0, np.pi * 2, 0.2)
    fixtures.append(("a", np.sin(angles) * 20 + 50, np.cos(angles) * 10 + 100))
    fixtures.append(("b", np.sin(angles) * 10 + 50, np.cos(angles) * 20 + 100))
    ha = np.linspace(0, np.pi, 10)
    fixtures.append(("half_circle", np.sin(ha) * 20 + 10, np.cos(ha) * 20 + 10))
    qa = np.linspace(0, np.pi / 2, 10)
    fixtures.append(("quarter_circle", np.sin(qa) * 20 + 10, np.cos(qa) * 20 + 10))
    fixtures.append((
        "real",
        np.array([0.06599215, 0.06105629, 0.06963708, 0.06900191, 0.06496001,
                  0.06352082, 0.05923421, 0.07080027, 0.07276284, 0.07170048]),
        np.array([0.05836343, 0.05866434, 0.05883284, 0.05872581, 0.05823667,
                  0.05839846, 0.0591999, 0.05907079, 0.05945377, 0.05909428]),
    ))
    for tag, pty, ptx in fixtures:
        pty = np.ascontiguousarray(pty.astype(np.float64))
        ptx = np.ascontiguousarray(ptx.astype(np.float64))
        x = ptx[:, None]
        y = pty[:, None]
        D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
        e = ellipse_mdl.fit_ellipse(pty, ptx)
        m = {
            "tag": tag,
            "pty": save(f"ellipse_{tag}_pty.npy", pty),
            "ptx": save(f"ellipse_{tag}_ptx.npy", ptx),
            "design": save(f"ellipse_{tag}_design.npy", D.astype(np.float64)),
            "params": {
                "center_1": float(e.center_1),
                "center_2": float(e.center_2),
                "angle": float(e.angle),
                "half_long_axis": float(e.half_long_axis),
                "half_short_axis": float(e.half_short_axis),
            },
        }
        ellipse_cases.append(m)
    manifest["ellipse"] = {"cases": ellipse_cases}

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", OUTDIR)
    print("label n:", [c["n"] for c in label_cases])
    print("watershed regions:", ws_meta["n_regions"])
    for c in blob_cases:
        print("blob localmax", c["tag"], "shape", c["shape"])
    print("ellipse cases:", [c["tag"] for c in ellipse_cases])


if __name__ == "__main__":
    main()
