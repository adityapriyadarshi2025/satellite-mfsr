"""Finding and loading SatSynthBurst-style bursts, and saving predictions.

numpy + opencv only, so every runner and the scorer share one loader.

On-disk layout (the one SuperF's data.py and its generator use):

    <root>/<scene>/scale_<df>_shift_<s>px_aug_<aug>/
        transform_log.json      # one entry per frame: path, true shifts, degradation
        sample_00.png ...       # LR frames, sample_00 is the base frame
        hr_ground_truth.png     # HR reference, aligned with sample_00

Predictions go to:

    <results>/<method>/<scene>/<config>/pred.npy   float32 HxWx3, RGB, unclipped
                                        meta.json  runtime, settings, notes
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

CONFIG_RE = re.compile(r"^scale_([0-9.]+)_shift_([0-9.]+)px_aug_([a-z]+)$")


def parse_scale(s):
    """'4' -> 4, '2.56' -> 2.56 (int when integral, so integer scales read 4, not 4.0, in folder names and tables)."""
    v = float(s)
    return int(v) if v.is_integer() else v


@dataclass(frozen=True)
class BurstRef:
    scene: str
    df: int | float      # upsampling factor; non-integer for the arbitrary-scale bursts (setup/make_arbscale.py)
    shift: str   # kept as the folder string, e.g. "1.0"
    aug: str
    path: str

    @property
    def config(self) -> str:
        return f"scale_{self.df}_shift_{self.shift}px_aug_{self.aug}"


def discover(root, df=None, shift=None, aug=None, scenes=None):
    """All bursts under root, optionally filtered. Sorted by scene then factor."""
    root = Path(root).resolve()
    found = []
    for log in root.rglob("transform_log.json"):
        m = CONFIG_RE.match(log.parent.name)
        if not m:
            continue
        ref = BurstRef(scene=log.parent.parent.name, df=parse_scale(m.group(1)),
                       shift=m.group(2), aug=m.group(3), path=str(log.parent))
        if df is not None and abs(float(ref.df) - float(df)) > 1e-9:
            continue
        if shift is not None and float(ref.shift) != float(shift):
            continue
        if aug is not None and ref.aug != aug:
            continue
        if scenes and ref.scene not in scenes:
            continue
        found.append(ref)
    return sorted(found, key=lambda r: (r.scene, r.df, r.shift, r.aug))


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / scale


def load_burst(ref: BurstRef | str, num_frames: int = 16):
    """Returns (frames [T,h,w,3] in [0,1], hr [H,W,3] in [0,1], transform_log).

    Frame order follows sorted transform_log keys, like SuperF (sample_00 first).
    """
    d = Path(ref.path if isinstance(ref, BurstRef) else ref)
    log = json.loads((d / "transform_log.json").read_text())
    keys = sorted(log.keys())[:num_frames]
    frames = np.stack([read_rgb(d / log[k]["path"]) for k in keys])
    hr = read_rgb(d / "hr_ground_truth.png")
    return frames, hr, {k: log[k] for k in keys}


def true_shifts_lr(log: dict):
    """[T,2] true (dx, dy) of each frame in LR pixels as recorded by the generator, or None if absent."""
    try:
        return np.array([[v["dx_pixels_lr"], v["dy_pixels_lr"]] for v in log.values()], dtype=np.float64)
    except (KeyError, TypeError):
        return None


def pred_dir(results_root, method: str, ref: BurstRef) -> Path:
    return Path(results_root) / method / ref.scene / ref.config


def save_prediction(results_root, method: str, ref: BurstRef, pred: np.ndarray, meta: dict):
    """pred: HxWx3 float RGB. Saved unclipped as float32 .npy plus an 8-bit preview."""
    out = pred_dir(results_root, method, ref)
    out.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(pred, dtype=np.float32)
    assert pred.ndim == 3 and pred.shape[2] == 3, pred.shape
    meta = {"method": method, "burst": asdict(ref), **meta}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    preview = (np.clip(pred, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(out / "pred_preview.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    tmp = out / "pred.npy.tmp"
    with open(tmp, "wb") as f:
        np.save(f, pred)
    os.replace(tmp, out / "pred.npy")   # pred.npy appears last and whole
    return out


def add_common_args(p):
    """CLI flags every runner shares."""
    p.add_argument("--data_root", default="data/satsynthburst", help="folder that holds the scene folders")
    p.add_argument("--results", default="results", help="where predictions go")
    p.add_argument("--df", type=parse_scale, nargs="+", default=[2, 4, 8], help="factors to run (non-integer allowed)")
    p.add_argument("--shift", default="1.0", help="shift variant folder value, e.g. 1.0")
    p.add_argument("--aug", default=None, help="aug variant (none/light/medium). default: whatever exists")
    p.add_argument("--scenes", nargs="*", default=None, help="only these scene folders")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--max_bursts", type=int, default=None, help="stop after N bursts (smoke tests)")
    p.add_argument("--skip_existing", action="store_true")
    return p


def select(args):
    refs = []
    for df in args.df:
        refs += discover(args.data_root, df=df, shift=args.shift, aug=args.aug, scenes=args.scenes)
    if not refs:
        raise SystemExit(f"no bursts found under {args.data_root} for df={args.df} shift={args.shift} aug={args.aug}")
    augs = sorted({r.aug for r in refs})
    if len(augs) > 1:
        raise SystemExit(f"several aug variants found {augs}; pick one with --aug")
    if args.max_bursts:
        refs = refs[: args.max_bursts]
    return refs
