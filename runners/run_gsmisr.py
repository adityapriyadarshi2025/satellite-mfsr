#!/usr/bin/env python3
"""Gaussian MISR v0 (our method, per-scene fit) on SatSynthBurst-style bursts.

    python runners/run_gsmisr.py --data_root data/satsynthburst --df 4 --aug light --scenes <scene>
    python runners/run_gsmisr.py --df 4 --aug light --tag gsmisr_lrbox --sub 1 --sub_box

Writes results/<tag>/<scene>/<config>/pred.npy + meta.json like every other runner, so
eval/evaluate.py and eval/report.py score it with the same protocol.
Model and options: gsmisr/misr.py (every Config field is a flag here).
"""

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.bursts import add_common_args, select, load_burst, save_prediction, pred_dir, true_shifts_lr  # noqa: E402
from gsmisr.misr import Config, fit, config_dict  # noqa: E402


def build_args():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--tag", default="gsmisr", help="results folder name for this variant")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    p.add_argument("--out_scale", type=float, default=None, help="output scale (default: the burst's df)")
    p.add_argument("--oracle_shifts", action="store_true",
                   help="diagnostic: fix every frame's shift at the generator's true value (not a method)")
    for f in fields(Config):
        name = "--" + f.name
        if f.type in ("bool", bool):
            p.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=f.default,
                           help=f"(default {f.default})")
        elif f.name in ("psf",):
            p.add_argument(name, type=float, default=f.default, help="LR px; default 1/df^2 (SatSynthBurst)")
        elif f.name in ("sub",):
            p.add_argument(name, type=int, default=f.default, help="default df (exact generator match)")
        else:
            typ = type(f.default) if f.default is not None else str
            p.add_argument(name, type=typ, default=f.default, help=f"(default {f.default})")
    return p.parse_args()


def main():
    a = build_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
    if a.oracle_shifts:
        cfg.init_shifts, cfg.learn_shifts, cfg.align_warmup = "given", False, 0
    for ref in select(a):
        if a.skip_existing and (pred_dir(a.results, a.tag, ref) / "pred.npy").exists():
            continue
        frames, hr, log = load_burst(ref, a.num_frames)
        t0 = time.time()
        res = fit(frames, ref.df, cfg, out_scale=a.out_scale, device=a.device,
                  given_d=true_shifts_lr(log) if a.oracle_shifts else None)
        total = time.time() - t0
        pred = res.hr
        if a.out_scale is None:
            assert pred.shape == hr.shape, (pred.shape, hr.shape)
        true = true_shifts_lr(log)
        meta = {"runtime_s": total, "fit_s": res.runtime_s, "ms_per_iter": res.ms_per_iter,
                "final_loss": res.final_loss, "config": config_dict(cfg), "device": a.device,
                "torch_threads": torch.get_num_threads(), "n_gaussians": int(res.model.mu0.shape[0]),
                "init_shifts_lr_px": res.init_shifts_lr_px, "final_shifts_lr_px": res.final_shifts_lr_px,
                "final_rot_rad": res.final_rot_rad, "noise_std_lr": res.noise_std,
                "true_shifts_lr_px": None if true is None else true.tolist(),
                "shift_convention": "final_shifts_lr_px = -b_t, same sign as the generator's dx/dy_pixels_lr",
                "out_scale": a.out_scale or ref.df, "oracle_shifts": a.oracle_shifts}
        out = save_prediction(a.results, a.tag, ref, pred, meta)
        print(f"{a.tag:12s} {ref.scene} x{ref.df}  {total:.0f}s ({res.ms_per_iter:.0f} ms/iter)  -> {out}", flush=True)


if __name__ == "__main__":
    main()
