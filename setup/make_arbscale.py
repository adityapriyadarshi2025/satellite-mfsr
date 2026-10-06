#!/usr/bin/env python3
"""SatSynthBurst-style bursts at any scale, including non-integer ones, from the same 20 HR images.

    python setup/make_arbscale.py --src data/satsynthburst --out data/arbscale            # all scales, none + light
    python setup/make_arbscale.py --src data/satsynthburst --out /tmp/x --check           # fidelity vs official bursts

Per scene:
  HR       hr_ground_truth.png of the official burst (256 x 256; identical across factors and augs)
  shifts   the official transform_log's dx/dy_pixels_lr. SatSynthBurst uses the same draws at every factor,
           so every scale here sees the same sub-pixel phases.
  frame t  HR shifted by d_t * s HR px (bilinear, edge replicate; the generator's output(x) = input(x + d))
           -> Gaussian blur, sigma = 1/s HR px -> exact area average to (256/s) x (256/s) (cv2.INTER_AREA;
           a plain block mean when s is an integer) -> 8-bit PNG.
  aug none   nothing else.
  aug light  our measurement of SatSynthBurst 'light': frames >= 2 get white noise
             N(0, 0.1^2) on the HR grid before the blur and a brightness/contrast change of up to 10 %;
             frames 0 and 1 untouched. An approximation of the official recipe, not a copy of it.
Scales: those with 256/s an integer, so the LR and HR grids line up exactly:
  2, 2.56, 3.2, 4, 5.12, 6.4, 8   ->   LR 128, 100, 80, 64, 50, 40, 32 px.
Borders differ a little from the official bursts (the official generator shifted the full image before
cropping; only the 256 crop is published). The 16 px evaluation crop removes that.

--check regenerates the integer scales with aug none and reports PSNR between our frames and the official
ones on the interior (2 LR px border removed). It should be at 8-bit quantisation level (> 50 dB).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.bursts import discover, read_rgb  # noqa: E402

SCALES = [2, 2.56, 3.2, 4, 5.12, 6.4, 8]
NOISE_HR = {"none": 0.0, "light": 0.10}
JITTER = {"none": 0.0, "light": 0.1}


def fmt(s) -> str:
    return str(int(s)) if float(s).is_integer() else repr(float(s))


def degrade(hr: np.ndarray, s: float, d_lr, rng=None, noise=0.0, jitter=0.0) -> np.ndarray:
    """hr [H,W,3] float in [0,1] -> LR frame [H/s, W/s, 3] float in [0,1]."""
    dx, dy = float(d_lr[0]) * s, float(d_lr[1]) * s
    x = nd_shift(hr, (-dy, -dx, 0), order=1, mode="nearest")
    if jitter > 0:
        b, c = rng.uniform(1 - jitter, 1 + jitter, 2)
        m = x.mean()
        x = np.clip((x - m) * c + m * b, 0, 1)
    if noise > 0:
        x = np.clip(x + rng.normal(0, noise, x.shape), 0, 1)
    x = gaussian_filter(x, sigma=(1.0 / s, 1.0 / s, 0), mode="reflect", truncate=4.0)
    h, w = int(round(hr.shape[0] / s)), int(round(hr.shape[1] / s))
    if float(s).is_integer():
        k = int(s)
        return x[: h * k, : w * k].reshape(h, k, w, k, 3).mean(axis=(1, 3))
    return cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)


def to8(x):
    return cv2.cvtColor((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def official_refs(src):
    """One official burst per scene (x4, aug none) to read the HR image and shifts from."""
    refs = discover(src, df=4, aug="none")
    if not refs:
        sys.exit(f"no official x4 'none' bursts under {src}")
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/satsynthburst", help="the official SatSynthBurst folder")
    ap.add_argument("--out", default="data/arbscale")
    ap.add_argument("--scales", type=float, nargs="+", default=SCALES)
    ap.add_argument("--augs", nargs="+", default=["none", "light"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="fidelity check against the official integer-scale bursts")
    a = ap.parse_args()

    refs = official_refs(a.src)
    if a.check:
        rows = []
        for ref in refs:
            hr = read_rgb(Path(ref.path) / "hr_ground_truth.png").astype(np.float64)
            log = json.loads((Path(ref.path) / "transform_log.json").read_text())
            keys = sorted(log)[: a.frames]
            for s in (2, 4, 8):
                off = discover(a.src, df=s, aug="none", scenes=[ref.scene])[0]
                for k in keys:
                    ours = degrade(hr, s, (log[k]["dx_pixels_lr"], log[k]["dy_pixels_lr"]))
                    ours = (np.clip(ours, 0, 1) * 255 + 0.5).astype(np.uint8).astype(np.float64) / 255
                    theirs = read_rgb(Path(off.path) / log[k]["path"]).astype(np.float64)
                    b = 2
                    mse = np.mean((ours[b:-b, b:-b] - theirs[b:-b, b:-b]) ** 2)
                    rows.append((s, 10 * np.log10(1 / max(mse, 1e-12))))
        for s in (2, 4, 8):
            v = np.array([p for ss, p in rows if ss == s])
            print(f"fidelity x{s}: PSNR ours vs official LR frames, interior: mean {v.mean():.1f} dB, min {v.min():.1f} dB "
                  f"over {len(v)} frames")
        return

    out_root = Path(a.out)
    for si, ref in enumerate(refs):
        hr = read_rgb(Path(ref.path) / "hr_ground_truth.png").astype(np.float64)
        log = json.loads((Path(ref.path) / "transform_log.json").read_text())
        keys = sorted(log)[: a.frames]
        for s in a.scales:
            for aug in a.augs:
                rng = np.random.default_rng([a.seed, si, int(round(s * 100)), a.augs.index(aug)])
                folder = out_root / ref.scene / f"scale_{fmt(s)}_shift_{ref.shift}px_aug_{aug}"
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / "hr_ground_truth.png"), to8(hr))
                newlog = {}
                for i, k in enumerate(keys):
                    d = (log[k]["dx_pixels_lr"], log[k]["dy_pixels_lr"])
                    noisy = aug != "none" and i >= 2
                    lr = degrade(hr, s, d, rng, noise=NOISE_HR[aug] if noisy else 0.0,
                                 jitter=JITTER[aug] if noisy else 0.0)
                    name = f"sample_{i:02d}.png"
                    cv2.imwrite(str(folder / name), to8(lr))
                    newlog[f"sample_{i:02d}"] = {
                        "dx_pixels_lr": float(d[0]), "dy_pixels_lr": float(d[1]),
                        "dx_pixels_hr": float(d[0]) * s, "dy_pixels_hr": float(d[1]) * s,
                        "dx_percent": float(d[0]) / lr.shape[1], "dy_percent": float(d[1]) / lr.shape[0],
                        "magnitude_pixels_lr": float(np.hypot(d[0], d[1])),
                        "magnitude_pixels_hr": float(np.hypot(d[0], d[1])) * s,
                        "shape": list(lr.shape), "path": name, "augmentation": aug,
                        "downsampling": {"method": "shift-bilinear, gaussian blur sigma 1/s HR px, area average "
                                                   "(mfsr/setup/make_arbscale.py)", "factor": s},
                        "noise_hr_std": NOISE_HR[aug] if noisy else 0.0,
                    }
                (folder / "transform_log.json").write_text(json.dumps(newlog, indent=1))
        print(f"{ref.scene}: {len(a.scales)} scales x {len(a.augs)} augs", flush=True)
    print(f"done -> {out_root}")


if __name__ == "__main__":
    main()
