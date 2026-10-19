#!/usr/bin/env python3
"""Progress report: PSNR per method x factor next to the paper's numbers, for each scoring
and dataset version, plus diagnostics that explain the numbers.

    python eval/report.py --results results > results/report.md

Reads results/scores.csv (code + paper protocol), and if present results/scores_compat.csv
(paper protocol, colormatch superf_compat) and results/scores_lpips.csv.
Diagnostics:
  * GNLL vs MSE on the same scenes (paper: +1.09 dB at x4)
  * alignment: learned shifts vs the generator's true shifts (slope, correlation)
  * s2-superres: does its output sit offset from the base frame? It learns the reference
    frame's shift too, so the whole solution can drift. Re-scored after shifting the
    prediction back by that drift, both signs.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from common.bursts import read_rgb  # noqa: E402
from metrics import score  # noqa: E402

PAPER = {  # SuperF (ICLR 2026) Table 1, SatSynthBurst PSNR
    "bilinear": {2: 34.69, 4: 29.71, 8: 26.62},
    "lafenetre": {2: 33.46, 4: 27.70, 8: 24.88},
    "nir": {2: 26.26, 4: 24.63, 8: 23.85},
    "superf_mse": {2: 36.73, 4: 32.94, 8: 28.87},
    "superf_gnll": {2: 37.26, 4: 34.03, 8: 29.28},
}
ORDER = ["bilinear", "bicubic", "gsasr_edsr", "gsasr_rdn", "lafenetre", "nir", "nir5k", "s2superres", "s2superres_asis",
         "superf_mse", "superf_mse_exact", "superf_gnll", "superf_gnll_gpu", "superf_gnll_tv", "gsmisr", "gsmisr_tv"]
# ablation rungs and frames-sweep runs get their own sections, not rows in the main tables
SECTION_ONLY = re.compile(r"^(gsmisr(_tv)?_(a\d+|n\d+)|superf_mse_n\d+|superf_gnll_tvp.*|gsmisr_tv_w.*|.*_h200)$")
LADDER = [  # jobs/ablation.sh: x4 light, 600 iterations; each step adds one ingredient
    ("gsmisr_a1", "s2-superres-like: fixed round Gaussians, phase-correlation shifts, no colour/noise model"),
    ("gsmisr_a2", "+ exhaustive shift search and warm-up"),
    ("gsmisr_a3", "+ per-frame colour model"),
    ("gsmisr_a4", "+ per-frame noise weights"),
    ("gsmisr_a5", "+ learnable positions"),
    ("gsmisr_a6", "+ learnable sizes (still round)"),
    ("gsmisr_a7", "+ anisotropic Gaussians, per-frame rotation (= full v0)"),
    ("gsmisr_tv", "+ TV prior on the HR image (= v0.1)"),
]
LADDER_SIDE = [  # compared with a7
    ("gsmisr_a8", "a7 with 1 sample per LR pixel (box approximation)"),
    ("gsmisr_a9", "a7 with the true shifts (upper bound, not a method)"),
]


def factors(df):
    """Factors present, sorted (2, 4, 8 for SatSynthBurst; also non-integer ones for the arbitrary-scale set)."""
    return sorted({float(v) for v in df.df.unique()}) if df is not None and len(df) else [2.0, 4.0, 8.0]


def fx(f) -> str:
    return f"{float(f):g}"


def load(path):
    p = Path(path)
    return pd.read_csv(p, dtype={"aug": str, "shift": str}) if p.exists() else None


def table(df, protocol, aug, title, metric="psnr", digits=2):
    sub = df[(df.protocol == protocol) & (df.aug == aug)] if df is not None else None
    if sub is None or sub.empty:
        return f"### {title}\n\n(no rows)\n"
    full = sub.groupby("df").scene.nunique().to_dict()
    methods = sorted((m for m in sub.method.unique() if not SECTION_ONLY.match(m)),
                     key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))
    if not methods:
        return ""
    fs = factors(sub)
    lines = [f"### {title}", "", "| method | " + " | ".join(f"x{fx(f)}" for f in fs) + " |",
             "|---|" + "---|" * len(fs)]
    for m in methods:
        cells = []
        for f in fs:
            s = sub[(sub.method == m) & (sub.df == f)][metric].dropna()
            if s.empty:
                cells.append("-")
                continue
            c = f"{s.mean():.{digits}f}"
            if len(s) < full.get(f, 0):
                c += f" (n={len(s)})"
            if metric == "psnr" and m in PAPER and float(f).is_integer() and int(f) in PAPER[m]:
                c += f" [{PAPER[m][int(f)]:.2f}]"
            cells.append(c)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paired(df, a, b, protocol="paper", aug="light"):
    """Per-factor mean of (a - b) over the scenes both have, plus how many scenes a wins."""
    out = []
    sub = df[(df.protocol == protocol) & (df.aug == aug)]
    for f in factors(sub):
        x = sub[(sub.method == a) & (sub.df == f)].set_index("scene").psnr
        y = sub[(sub.method == b) & (sub.df == f)].set_index("scene").psnr
        common = x.index.intersection(y.index)
        if len(common):
            d = x[common] - y[common]
            out.append(f"x{fx(f)}: {a} - {b} = {d.mean():+.2f} dB on {len(common)} scenes "
                       f"({a} better on {int((d > 0).sum())}/{len(common)}; range {d.min():+.2f} to {d.max():+.2f})")
    return out


def per_scene(df, methods, f, protocol="paper", aug="light"):
    """Markdown table of PSNR per scene for one factor."""
    sub = df[(df.protocol == protocol) & (df.aug == aug) & (df.df == f)]
    piv = sub.pivot_table(index="scene", columns="method", values="psnr")
    cols = [m for m in methods if m in piv.columns]
    if piv.empty or len(cols) < 2:
        return ""
    lines = [f"### Per scene, x{fx(f)}, `{aug}` (paper protocol)", "",
             "| scene | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for scene, row in piv[cols].iterrows():
        best = row.max()
        cells = [("**%.2f**" % v if v == best else "%.2f" % v) if v == v else "-" for v in row]
        lines.append(f"| {scene[:30]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ladder(sc, lp):
    """Ablation ladder at x4 light: mean PSNR per rung and the paired gain of each step."""
    sub = sc[(sc.protocol == "paper") & (sc.aug == "light") & (sc.df == 4)]
    rungs = [m for m, _ in LADDER + LADDER_SIDE]
    if not any(m.startswith("gsmisr_a") and m in set(sub.method) for m in rungs):
        return ""
    lpm = {}
    if lp is not None:
        l4 = lp[(lp.aug == "light") & (lp.df == 4)]
        lpm = l4.groupby("method").lpips.mean().to_dict()
    psnr = {m: sub[sub.method == m].set_index("scene").psnr for m in rungs + ["superf_gnll", "gsmisr"]}
    lines = ["## Ablation ladder (x4, `light`, 600 iterations, paper protocol)", "",
             "| step | method | what changes | PSNR | gain of this step (paired) | LPIPS |",
             "|---|---|---|---|---|---|"]
    prev = None
    for i, (m, what) in enumerate(LADDER + LADDER_SIDE):
        x = psnr[m]
        base = "gsmisr_a7" if (m, what) in LADDER_SIDE else prev
        if x.empty:
            lines.append(f"| {i + 1} | {m} | {what} | - | - | - |")
            continue
        if (m, what) in LADDER:
            prev = m          # the next step is compared with the last rung that has results
        cell = f"{x.mean():.2f}" + (f" (n={len(x)})" if len(x) < 20 else "")
        gain = "-"
        if base is not None and not psnr[base].empty:
            c = x.index.intersection(psnr[base].index)
            if len(c):
                d = x[c] - psnr[base][c]
                gain = f"{d.mean():+.2f} vs {base.replace('gsmisr_', '')} (better on {int((d > 0).sum())}/{len(c)})"
        lp_cell = f"{lpm[m]:.4f}" if m in lpm and lpm[m] == lpm[m] else "-"
        lines.append(f"| {i + 1} | {m} | {what} | {cell} | {gain} | {lp_cell} |")
    refs = [f"{m} {psnr[m].mean():.2f}" for m in ("superf_gnll", "gsmisr") if not psnr[m].empty]
    if refs:
        lines += ["", "For reference at x4 `light`: " + ", ".join(refs) + " (gsmisr = v0 at 1000 iterations)."]
    return "\n".join(lines) + "\n"


def frames_sweep(sc):
    """PSNR vs number of input frames on clean bursts, with the random-shift coverage prediction."""
    sub = sc[(sc.protocol == "paper") & (sc.aug == "none")]
    fams = []
    for prefix, label in (("gsmisr_tv_n", "TV version"), ("gsmisr_n", "v0, no TV"), ("superf_mse_n", "SuperF MSE")):
        ns = sorted({int(m[len(prefix):]) for m in sub.method.unique()
                     if m.startswith(prefix) and m[len(prefix):].isdigit()})
        if prefix == "superf_mse_n" and ns and "superf_mse" in set(sub.method):
            ns = sorted(set(ns) | {16})            # its 16-frame run is plain superf_mse
        if ns:
            fams.append((prefix, label, ns))
    if not fams:
        return ""
    fs = [f for f in (4.0, 8.0) if (sub.df == f).any()]
    allns = sorted({n for _, _, ns in fams for n in ns})
    head = ["frames"] + [f"{label} x{fx(f)}" for _, label, _ in fams for f in fs] + \
           [f"coverage x{fx(f)} (predicted)" for f in fs]
    lines = ["## Frames sweep: PSNR vs number of frames, `none` (paper protocol)", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in allns:
        cells = [str(n)]
        for prefix, _, _ in fams:
            for f in fs:
                name = f"{prefix}{n}"
                if prefix == "superf_mse_n" and n == 16 and name not in set(sub.method):
                    name = "superf_mse"
                s = sub[(sub.method == name) & (sub.df == f)].psnr
                cells.append("-" if s.empty else f"{s.mean():.2f}" + (f" (n={len(s)})" if len(s) < 20 else ""))
        cells += [f"{1 - (1 - 1 / f ** 2) ** n:.2f}" for f in fs]
        lines.append("| " + " | ".join(cells) + " |")
    bic = [f"x{fx(f)} {sub[(sub.method == 'bicubic') & (sub.df == f)].psnr.mean():.2f}" for f in fs
           if not sub[(sub.method == "bicubic") & (sub.df == f)].empty]
    if bic:
        lines += ["", "Bicubic on frame 0 alone (`none`): " + ", ".join(bic) + ". Coverage = share of the "
                  "s x s sub-pixel phases hit by at least one of N randomly shifted frames, 1 - (1 - 1/s^2)^N."]
    return "\n".join(lines) + "\n"


def tv_weights(sc, res):
    """Our TV weight check (x4/x8 light) with the per-scene best, and the SuperF+TV pilot."""
    sub = sc[(sc.protocol == "paper") & (sc.aug == "light")]
    ours = {"gsmisr_tv_w3e-4": 3e-4, "gsmisr_tv": 1e-3, "gsmisr_tv_w3e-3": 3e-3}
    lines = []
    for f in (4.0, 8.0):
        piv = sub[sub.df == f].pivot_table(index="scene", columns="method", values="psnr")
        cols = [m for m in ours if m in piv.columns]
        if len(cols) < 2:
            continue
        p = piv[cols].dropna()
        cells = [f"{m} ({ours[m]:g}): {p[m].mean():.2f}" for m in cols]
        best = p.max(axis=1)
        pick = p.idxmax(axis=1).map(lambda m: f"{ours[m]:g}").value_counts().to_dict()
        lines.append(f"- x{fx(f)} on {len(p)} scenes: " + ", ".join(cells)
                     + f"; best weight per scene (oracle): {best.mean():.2f} (+{best.mean() - p['gsmisr_tv'].mean():.2f}"
                     f" over 1e-3; picks {pick})" if "gsmisr_tv" in cols else "")
    pil = sub[(sub.df == 4) & sub.method.str.startswith("superf_gnll_tvp")]
    if len(pil):
        g = pil.groupby("method").psnr.agg(["mean", "size"])
        g.index = [m.replace("superf_gnll_tvp", "") for m in g.index]
        g = g.sort_index(key=lambda i: i.astype(float))
        lines.append("- SuperF GNLL + TV pilot (x4, the 2 scenes our weight was tuned on): "
                     + ", ".join(f"{w}: {r['mean']:.2f}" + (f" (n={int(r['size'])})" if r["size"] < 2 else "")
                                 for w, r in g.iterrows()))
    wf = Path(res) / "superf_tv_weight.txt"
    if wf.exists():
        lines.append(f"- SuperF + TV weight used: {wf.read_text().strip()}")
    lines = [l for l in lines if l]
    if not lines:
        return ""
    return "## TV weight and the SuperF + TV control\n\n" + "\n".join(lines) + "\n"


def shift_fit(est, true):
    """slope and correlation of estimated vs true shifts, frames 1..T-1, both axes pooled."""
    e, t = np.asarray(est, float)[1:].ravel(), np.asarray(true, float)[1:].ravel()
    if len(e) < 2 or np.std(t) == 0:
        return np.nan, np.nan
    e, t = e - e.mean(), t - t.mean()
    return float((e * t).sum() / (t * t).sum()), float(np.corrcoef(e, t)[0, 1])


def alignment(results: Path, method: str, aug: str, df_: int):
    slopes, corrs, drifts, rows = [], [], [], []
    for meta_path in sorted(results.glob(f"{method}/*/scale_{df_}_shift_*_aug_{aug}/meta.json")):
        m = json.loads(meta_path.read_text())
        true = m.get("true_shifts_lr_px")
        if method.startswith("superf"):
            est = m.get("learned_shift_params")
            if est is None or true is None:
                continue
            lr_w = 256 // df_
            est = np.asarray(est, float) * lr_w          # coordinate units -> LR px
        else:
            est = m.get("final_shifts_lr_px")
            if est is None or true is None:
                continue
            est = np.asarray(est, float)
            drifts.append(est[0])
            rows.append((meta_path.parent, est[0]))
        s, c = shift_fit(est, true)
        slopes.append(s)
        corrs.append(c)
    return slopes, corrs, drifts, rows


def s2_drift_test(rows, df_):
    """Re-score s2-superres after moving the prediction by +/- (frame-0 shift x df) HR px."""
    base, plus, minus = [], [], []
    for d, (dx, dy) in rows:
        meta = json.loads((d / "meta.json").read_text())
        gt = read_rgb(Path(meta["burst"]["path"]) / "hr_ground_truth.png")
        pred = np.load(d / "pred.npy")
        base.append(score(pred, gt, "paper")["psnr"])
        for sign, acc in ((+1, plus), (-1, minus)):
            moved = nd_shift(pred, (sign * dy * df_, sign * dx * df_, 0), order=3, mode="nearest")
            acc.append(score(moved, gt, "paper")["psnr"])
    return np.mean(base), np.mean(plus), np.mean(minus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    res = Path(a.results)
    sc, compat, lp = load(res / "scores.csv"), load(res / "scores_compat.csv"), load(res / "scores_lpips.csv")
    if sc is None:
        sys.exit("no results/scores.csv yet")
    augs = sorted(sc.aug.unique())
    print("# MFSR baseline report\n")
    print("PSNR in dB, mean over scenes; the SuperF paper's number in [brackets]; (n=) when a "
          "column isn't finished yet. Paper protocol = DBSR colour match + 16 px crop (what the "
          "paper says it did). Code protocol = what SuperF's optimize.py prints.\n")
    for aug in augs:
        print(table(sc, "paper", aug, f"Paper protocol, `{aug}`"))
    if "light" in augs:
        print(table(sc, "code", "light", "Code protocol, `light`"))
        if compat is not None:
            print(table(compat, "paper", "light", "Paper protocol, superf_compat colour match, `light`"))
        if lp is not None and not lp.lpips.isna().all():
            print(table(lp, "paper", "light", "LPIPS (AlexNet, lower is better), `light`", metric="lpips", digits=4))

    print("## Diagnostics\n")
    bl = sc[(sc.protocol == "paper") & (sc.method == "bilinear")]
    for f in (2, 4, 8):
        s = bl[(bl.df == f)].groupby("aug").psnr.mean()
        if len(s) and f in PAPER["bilinear"]:
            print(f"- bilinear x{f}: ours " + ", ".join(f"{v:.2f} [{k}]" for k, v in s.items())
                  + f" vs paper {PAPER['bilinear'][f]:.2f} (same base frame in every version)")
    for line in paired(sc, "superf_gnll", "superf_mse"):
        print(f"- {line} (paper x4: +1.09)")
    for line in paired(sc, "bicubic", "superf_mse"):
        print(f"- {line}")
    for other in ("superf_gnll", "superf_mse", "bicubic", "s2superres"):
        for line in paired(sc, "gsmisr", other):
            print(f"- {line}")
    for aug in augs:   # the TV version (v0.1); on results_arb also the non-integer scales
        for other in ("superf_gnll_tv", "superf_gnll", "superf_gnll_gpu", "superf_mse", "superf_mse_exact", "bicubic",
                      "gsmisr"):
            for line in paired(sc, "gsmisr_tv", other, aug=aug):
                print(f"- [{aug}] {line}")
    # What the prior alone does for SuperF. superf_gnll_gpu is plain SuperF GNLL run on the GPU like
    # superf_gnll_tv, so that pair differs only by the prior; vs the CPU superf_gnll it shows
    # the CPU/GPU difference.
    for a, b in (("superf_gnll_tv", "superf_gnll_gpu"), ("superf_gnll_tv", "superf_gnll"),
                 ("superf_gnll_gpu", "superf_gnll")):
        for line in paired(sc, a, b):
            print(f"- {line}")
    for aug in augs:   # SuperF with an exact pixel-average downsampler vs its own 'area' pooling (results_arb)
        for line in paired(sc, "superf_mse_exact", "superf_mse", aug=aug):
            print(f"- [{aug}] {line}")
    if compat is not None:
        for m in ("superf_mse", "superf_gnll", "bilinear", "bicubic"):
            x = sc[(sc.protocol == "paper") & (sc.aug == "light") & (sc.method == m) & (sc.df == 4)].set_index("scene").psnr
            y = compat[(compat.aug == "light") & (compat.method == m) & (compat.df == 4)].set_index("scene").psnr
            c = x.index.intersection(y.index)
            if len(c):
                print(f"- superf_compat colour match changes {m} x4 by {(y[c] - x[c]).mean():+.2f} dB")
    for method in ("superf_mse", "s2superres", "gsmisr", "gsmisr_tv"):
        for aug in augs:
            slopes, corrs, drifts, rows = alignment(res, method, aug, 4)
            if not slopes:
                continue
            msg = (f"- {method} x4 [{aug}] alignment vs true shifts: slope {np.nanmean(slopes):+.2f}, "
                   f"corr {np.nanmean(corrs):+.2f} over {len(slopes)} bursts")
            if drifts:
                dr = np.abs(np.asarray(drifts)).mean(axis=0)
                msg += f"; reference-frame drift |dx|,|dy| = {dr[0]:.3f}, {dr[1]:.3f} LR px ({dr[0] * 4:.2f}, {dr[1] * 4:.2f} HR px)"
            print(msg)
            if method == "s2superres" and rows:
                b, p, m_ = s2_drift_test(rows, 4)
                print(f"  - s2-superres x4 [{aug}] re-scored after undoing the drift: as saved {b:.2f}, "
                      f"moved + {p:.2f}, moved - {m_:.2f} dB")
    if {"gsmisr", "gsmisr_tv"} & set(sc.method):
        print()
        for f in factors(sc):
            tab = per_scene(sc, ["bicubic", "s2superres", "superf_mse", "superf_gnll", "superf_gnll_gpu",
                                 "superf_gnll_tv", "gsmisr", "gsmisr_tv"], f)
            if tab:
                print(tab)
    for section in (ladder(sc, lp), tv_weights(sc, res), frames_sweep(sc)):
        if section:
            print(section)
    rt = sc[sc.protocol == "paper"].groupby("method").runtime_s.apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    print("- mean runtime per burst (s, 8 CPU threads): " + ", ".join(f"{k} {v:.0f}" for k, v in rt.items() if v == v))


if __name__ == "__main__":
    main()
