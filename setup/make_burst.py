#!/usr/bin/env python3
"""Make SatSynthBurst-style bursts from any HR image (numpy/scipy only).

Uses: smoke tests without the real dataset, and new variants the real dataset
doesn't have, e.g. a realistic sensor blur (--psf_hr_px 2.0 at x4) or more frames.
The real SatSynthBurst comes from Zenodo (setup/get_satsynthburst.py); this is
a close re-implementation of SuperF's generator, not bit-identical:

  SuperF generator (create_data_from_single_image.py + utils.py):
    shift the full HR image by (dx, dy) HR px with bilinear grid_sample
    (dx, dy uniform in [-shift*df, +shift*df], frame 0 unshifted)
    -> crop the HR patch -> Gaussian blur sigma = 1/df HR px -> df x df average pool
    -> aug 'light'/'medium': brightness/contrast/saturation/hue jitter on frames >= 2
       and Gaussian noise (0.005 / 0.01) on frames >= 1, applied before downsampling
  Here: same geometry and degradation; colour jitter is brightness/contrast only.

    python setup/make_burst.py --hr path/to/hr.png --out data/smoke --df 2 4 8
    python setup/make_burst.py --hr_dir data/hr_images --out data/realblur --df 4 --psf_hr_px 2.0
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

NOISE = {"none": 0.0, "light": 0.005, "medium": 0.01, "heavy": 0.02}
JITTER = {"none": 0.0, "light": 0.1, "medium": 0.2, "heavy": 0.4}


def degrade(hr_patch, df, psf_hr_px):
    x = gaussian_filter(hr_patch, sigma=(psf_hr_px, psf_hr_px, 0), mode="reflect", truncate=4.0) if psf_hr_px > 0 else hr_patch
    h, w = x.shape[0] // df, x.shape[1] // df
    return x[: h * df, : w * df].reshape(h, df, w, df, 3).mean(axis=(1, 3))


def make(hr_full, name, out_root, df, shift, n, aug, patch, psf_hr_px, seed, border_crop):
    rng = np.random.default_rng(seed)
    img = hr_full[border_crop:-border_crop, border_crop:-border_crop] if border_crop else hr_full
    H, W = img.shape[:2]
    if patch:
        cy, cx = H // 2, W // 2
        y0, x0 = cy - patch // 2, cx - patch // 2
        if y0 < 0 or x0 < 0 or y0 + patch > H or x0 + patch > W:
            raise SystemExit(f"{name}: image {W}x{H} too small for patch {patch}")
        sl = (slice(y0, y0 + patch), slice(x0, x0 + patch))
    else:
        sl = (slice(0, H - H % df), slice(0, W - W % df))
    folder = Path(out_root) / name / f"scale_{df}_shift_{shift:.1f}px_aug_{aug}"
    folder.mkdir(parents=True, exist_ok=True)
    hr_patch = img[sl]
    cv2.imwrite(str(folder / "hr_ground_truth.png"),
                cv2.cvtColor((np.clip(hr_patch, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    sigma = (1.0 / df) if psf_hr_px is None else psf_hr_px
    log = {}
    for i in range(n):
        dx, dy = (0.0, 0.0) if i == 0 else tuple(rng.uniform(-shift * df, shift * df, 2))
        # SuperF convention: output(x) = input(x + dx)
        moved = nd_shift(img, (-dy, -dx, 0), order=1, mode="nearest")[sl]
        if aug != "none" and i > 0:
            if i > 1 and JITTER[aug] > 0:
                b, c = rng.uniform(1 - JITTER[aug], 1 + JITTER[aug], 2)
                m = moved.mean()
                moved = np.clip((moved - m) * c + m * b, 0, 1)
            moved = np.clip(moved + rng.normal(0, NOISE[aug], moved.shape), 0, 1)
        lr = degrade(moved, df, sigma)
        cv2.imwrite(str(folder / f"sample_{i:02d}.png"),
                    cv2.cvtColor((np.clip(lr, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        log[f"sample_{i:02d}"] = {
            "dx_pixels_hr": float(dx), "dy_pixels_hr": float(dy),
            "dx_pixels_lr": float(dx / df), "dy_pixels_lr": float(dy / df),
            "dx_percent": float(dx / df / lr.shape[1]), "dy_percent": float(dy / df / lr.shape[0]),
            "shape": list(lr.shape), "path": f"sample_{i:02d}.png", "augmentation": aug,
            "downsampling": {"method": "gaussian_blur_avg_pool (mfsr/setup/make_burst.py)", "factor": df,
                             "gaussian_sigma_pixels": sigma, "window_size_pixels": df},
        }
    (folder / "transform_log.json").write_text(json.dumps(log, indent=2))
    return folder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hr", nargs="+", help="HR image file(s)")
    src.add_argument("--hr_dir", help="folder of HR images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--df", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--shift", type=float, default=1.0, help="max shift in LR px (SuperF default 1.0)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--aug", default="none", choices=list(NOISE))
    ap.add_argument("--patch", type=int, default=256, help="HR patch size (0 = whole image)")
    ap.add_argument("--border_crop", type=int, default=20)
    ap.add_argument("--psf_hr_px", type=float, default=None, help="blur sigma in HR px (default 1/df, as SatSynthBurst)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = [Path(f) for f in a.hr] if a.hr else sorted(p for p in Path(a.hr_dir).iterdir()
                                                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    for k, f in enumerate(files):
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        for df in a.df:
            out = make(img, f.stem, a.out, df, a.shift, a.frames, a.aug, a.patch, a.psf_hr_px, a.seed + k, a.border_crop)
            print(out)


if __name__ == "__main__":
    main()
