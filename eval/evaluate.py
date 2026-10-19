#!/usr/bin/env python3
"""One scorer for every method. Reads results/<method>/<scene>/<config>/pred.npy,
scores it against that burst's hr_ground_truth.png, writes results/scores.csv.

    python eval/evaluate.py --data_root data/satsynthburst --protocol paper
    python eval/evaluate.py --protocol both --lpips --device cuda
    python eval/evaluate.py --methods bilinear superf_mse --protocol code

Then:  python eval/make_table.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common.bursts import read_rgb  # noqa: E402
from metrics import score  # noqa: E402

FIELDS = ["method", "scene", "df", "shift", "aug", "protocol", "psnr", "ssim", "lpips",
          "runtime_s", "ssim_impl", "colormatch"]


def find_preds(results: Path, methods):
    for pred in sorted(results.glob("*/*/*/pred.npy")):
        method = pred.parts[-4]
        if methods and method not in methods:
            continue
        yield method, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/satsynthburst")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--protocol", choices=["code", "paper", "both"], default="paper")
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs torch + lpips weights)")
    ap.add_argument("--colormatch", choices=["dbsr", "superf_compat", "dbsr_affine"], default="dbsr",
                    help="dbsr_affine (3x3 + offset) for real data, e.g. WorldStrat")
    ap.add_argument("--border", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="csv path (default results/scores.csv)")
    args = ap.parse_args()

    results = Path(args.results)
    protocols = ["code", "paper"] if args.protocol == "both" else [args.protocol]
    rows = []
    for method, pred_path in find_preds(results, args.methods):
        try:
            meta = json.loads((pred_path.parent / "meta.json").read_text())
            b = meta["burst"]
            burst_dir = Path(b["path"])
            if not burst_dir.exists():  # moved machines: rebuild from data_root
                burst_dir = Path(args.data_root) / b["scene"] / pred_path.parent.name
            gt = read_rgb(burst_dir / "hr_ground_truth.png")
            pred = np.load(pred_path)
        except Exception as e:  # noqa: BLE001  (e.g. a file another job is writing right now)
            print(f"skipping {pred_path}: {e}")
            continue
        for prot in protocols:
            m = score(pred, gt, prot, use_lpips=args.lpips, colormatch=args.colormatch,
                      border=args.border, device=args.device)
            rows.append({"method": method, "scene": b["scene"], "df": b["df"], "shift": b["shift"],
                         "aug": b["aug"], "protocol": prot, **m, "runtime_s": meta.get("runtime_s", ""),
                         "colormatch": args.colormatch if prot == "paper" else ""})
            print(f"{method:14s} {b['scene'][:28]:28s} x{b['df']} {prot:5s} "
                  f"PSNR {m['psnr']:6.2f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}")

    out = Path(args.out) if args.out else results / "scores.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
