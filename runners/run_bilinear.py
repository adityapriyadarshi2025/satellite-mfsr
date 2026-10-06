#!/usr/bin/env python3
"""Bilinear (and bicubic) upsampling of the base frame. The must-have row.

Same call SuperF uses for its own bilinear number: cv2.resize(INTER_LINEAR) on
sample_00 as float in [0,1]. numpy/opencv only, runs anywhere.

    python runners/run_bilinear.py --data_root data/satsynthburst --df 2 4 8
    python runners/run_bilinear.py --interp bicubic
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bursts import add_common_args, select, load_burst, save_prediction, pred_dir  # noqa: E402

INTERP = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--interp", choices=list(INTERP), default="bilinear")
    p.add_argument("--device", default="cpu", help="ignored (accepted so the shared job scripts can pass it)")
    args = p.parse_args()
    for ref in select(args):
        if args.skip_existing and (pred_dir(args.results, args.interp, ref) / "pred.npy").exists():
            continue
        frames, hr, _ = load_burst(ref, args.num_frames)
        t0 = time.time()
        pred = cv2.resize(frames[0], (hr.shape[1], hr.shape[0]), interpolation=INTERP[args.interp])
        dt = time.time() - t0
        out = save_prediction(args.results, args.interp, ref, pred,
                              {"runtime_s": dt, "input": "base frame only (sample_00)"})
        print(f"{args.interp:9s} {ref.scene} x{ref.df} -> {out}")


if __name__ == "__main__":
    main()
