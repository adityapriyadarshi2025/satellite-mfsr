#!/usr/bin/env python3
"""Helpers for inspecting a SatSynthBurst folder (stdlib + common.bursts; pandas for 'pick').

    python jobs/variants.py --data_root data/satsynthburst report
    python jobs/variants.py --data_root data/satsynthburst shift            -> 1.0 if present
    python jobs/variants.py --data_root data/satsynthburst augs --shift 1.0 -> one per line, likeliest first
    python jobs/variants.py --data_root D first-scene --df 4 --aug light --shift 1.0
    python jobs/variants.py --data_root D scene-parent --df 4 --aug light --shift 1.0 --scene X
    python jobs/variants.py pick --scores results/scores.csv               -> aug closest to the paper
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bursts import discover  # noqa: E402

# The paper uses "spectral variations and additive Gaussian noise in all experiments", so
# augmented variants are tried first.
AUG_ORDER = ["light", "medium", "none", "heavy"]
PAPER = {"superf_mse": {2: 36.73, 4: 32.94, 8: 28.87}, "bilinear": {2: 34.69, 4: 29.71, 8: 26.62}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/satsynthburst")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    sub.add_parser("shift")
    a = sub.add_parser("augs"); a.add_argument("--shift", default="1.0")
    for name in ("first-scene", "scene-parent"):
        s = sub.add_parser(name)
        s.add_argument("--df", type=int, default=4)
        s.add_argument("--aug", required=True)
        s.add_argument("--shift", default="1.0")
        s.add_argument("--scene", default=None)
    p = sub.add_parser("pick"); p.add_argument("--scores", default="results/scores.csv")
    args = ap.parse_args()

    if args.cmd == "pick":
        # The base frame (and so bilinear) is identical across aug variants, so compare the
        # SuperF-minus-bilinear gain per scene with the paper's (32.94 - 29.71 = 3.23 dB at x4).
        # That cancels scene difficulty and works on a subset of scenes too.
        import pandas as pd
        df = pd.read_csv(args.scores, dtype={"aug": str})
        df = df[(df.protocol == "paper") & (df.df == 4)]
        sf = df[df.method == "superf_mse"].set_index(["aug", "scene"]).psnr
        bl = df[df.method == "bilinear"].set_index(["aug", "scene"]).psnr
        gain = (sf - bl).dropna()
        if gain.empty:
            sys.exit(1)
        target = PAPER["superf_mse"][4] - PAPER["bilinear"][4]
        if not bl.empty:
            print(f"  bilinear x4 (paper protocol): {bl.mean():.2f} dB, paper 29.71", file=sys.stderr)
        per_aug = gain.groupby(level=0).agg(["mean", "count"])
        for aug, row in per_aug.iterrows():
            m = sf.loc[aug].mean()
            print(f"  [{aug}] superf_mse x4 {m:.2f} dB, gain over bilinear {row['mean']:+.2f} dB "
                  f"on {int(row['count'])} scenes (paper +{target:.2f})", file=sys.stderr)
        print((per_aug["mean"] - target).abs().idxmin())
        return

    refs = discover(args.data_root)
    if args.cmd == "report":
        if not refs:
            print(f"  no bursts under {args.data_root}")
            return
        print(f"  {len(refs)} bursts, {len({r.scene for r in refs})} scenes under {Path(args.data_root).resolve()}")
        for (df, sh, aug), n in sorted(Counter((r.df, r.shift, r.aug) for r in refs).items()):
            print(f"    x{df} shift {sh} aug {aug}: {n} bursts")
        r = refs[0]
        log = json.loads((Path(r.path) / "transform_log.json").read_text())
        k = sorted(log)[min(1, len(log) - 1)]
        print(f"  frames per burst: {len(log)}; example entry keys: {sorted(log[k])}")
        return
    if args.cmd == "shift":
        shifts = sorted({r.shift for r in refs}, key=float)
        print("1.0" if "1.0" in shifts else (shifts[0] if shifts else "1.0"))
        return
    if args.cmd == "augs":
        augs = {r.aug for r in refs if float(r.shift) == float(args.shift)}
        for aug in sorted(augs, key=lambda x: AUG_ORDER.index(x) if x in AUG_ORDER else 99):
            print(aug)
        return
    sel = [r for r in refs if r.df == args.df and r.aug == args.aug and float(r.shift) == float(args.shift)
           and (args.scene is None or r.scene == args.scene)]
    if not sel:
        sys.exit(1)
    if args.cmd == "first-scene":
        print(sel[0].scene)
    else:
        print(Path(sel[0].path).parent.parent)


if __name__ == "__main__":
    main()
