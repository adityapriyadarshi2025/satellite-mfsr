#!/usr/bin/env python3
"""Turn results/scores.csv into the baseline table (mean (std) over scenes).

    python eval/make_table.py                   # paper protocol, PSNR/SSIM/LPIPS
    python eval/make_table.py --protocol code

Writes results/table_<protocol>.md. Cells show n=<scenes> when a method has not
finished every scene, so a half-done row can't pass as a full one. The SuperF
paper's own Table 1 PSNR numbers are appended for reference only: they came from
their runs, not ours.
"""

import argparse
from pathlib import Path

import pandas as pd

# SuperF (ICLR 2026) Table 1, SatSynthBurst PSNR mean (std), 16 frames. Copied from the paper PDF.
PAPER_PSNR = {
    "Bilinear (paper)": {2: "34.69 (3.50)", 4: "29.71 (3.64)", 8: "26.62 (3.68)"},
    "Lafenetre et al. 2023 (paper)": {2: "33.46 (3.62)", 4: "27.70 (3.79)", 8: "24.88 (3.71)"},
    "NIR [2k] (paper)": {2: "26.26 (3.91)", 4: "24.63 (4.41)", 8: "23.85 (3.79)"},
    "NIR [5k] (paper)": {2: "25.65 (5.82)", 4: "24.99 (4.12)", 8: "23.61 (2.97)"},
    "SuperF MSE [2k] (paper)": {2: "36.73 (1.66)", 4: "32.94 (1.83)", 8: "28.87 (2.32)"},
    "SuperF GNLL [2k] (paper)": {2: "37.26 (2.30)", 4: "34.03 (2.71)", 8: "29.28 (3.21)"},
}
ORDER = ["bilinear", "bicubic", "lafenetre", "nir", "s2superres", "s2superres_asis",
         "superf_mse", "superf_gnll"]


def fmt(s: pd.Series, digits: int, n_max: int) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    cell = f"{s.mean():.{digits}f} ({s.std(ddof=0):.{digits}f})"
    return cell if len(s) == n_max else f"{cell} n={len(s)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--protocol", default="paper")
    ap.add_argument("--aug", default=None, help="only this aug variant (recommended when several exist)")
    ap.add_argument("--shift", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.scores, dtype={"aug": str, "shift": str})
    df = df[df.protocol == args.protocol]
    if args.aug:
        df = df[df.aug == args.aug]
    if args.shift:
        df = df[df["shift"].astype(float) == float(args.shift)]
    if df.empty:
        raise SystemExit(f"no rows for protocol={args.protocol} aug={args.aug} shift={args.shift}")
    factors = sorted(df.df.unique())
    n_max = df.groupby("df").scene.nunique().to_dict()
    methods = sorted(df.method.unique(), key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))

    lines = [f"# Baseline table ({args.protocol} protocol)", "",
             f"aug variant(s): {', '.join(sorted(df.aug.astype(str).unique()))}; "
             f"shift: {', '.join(sorted(df['shift'].astype(str).unique()))}; "
             f"scenes per factor: {n_max}", ""]
    for metric, digits, arrow in [("psnr", 2, "higher"), ("ssim", 4, "higher"), ("lpips", 4, "lower")]:
        if df[metric].isna().all():
            continue
        lines += [f"## {metric.upper()} ({arrow} is better)", "",
                  "| method | " + " | ".join(f"x{f}" for f in factors) + " | runtime s (mean) |",
                  "|---|" + "---|" * len(factors) + "---|"]
        for m in methods:
            sub = df[df.method == m]
            cells = [fmt(sub[sub.df == f][metric], digits, n_max.get(f, 0)) for f in factors]
            rt = pd.to_numeric(sub.runtime_s, errors="coerce").mean()
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {rt:.1f} |")
        if metric == "psnr":
            for name, vals in PAPER_PSNR.items():
                lines.append(f"| *{name}* | " + " | ".join(vals.get(int(f), "-") for f in factors) + " | |")
        lines.append("")

    suffix = f"_{args.aug}" if args.aug else ""
    out = Path(args.scores).parent / f"table_{args.protocol}{suffix}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
