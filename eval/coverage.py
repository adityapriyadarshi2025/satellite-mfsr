#!/usr/bin/env python3
"""How well each burst's frames cover the sub-pixel positions, and whether that explains
per-scene gains. No new runs needed: shifts come from transform_log.json, gains from scores.csv.

    python eval/coverage.py --data_root data/satsynthburst --results results --aug light

For a burst with true LR shifts d_t, the sampling phases are d_t mod 1 (LR px). At factor k the
unit cell splits into k x k sub-cells; coverage = fraction of sub-cells holding at least one
phase. For N random shifts the expectation is 1 - (1 - 1/k^2)^N.
Also reported: the largest hole (fill distance on the torus, LR px) and each scene's HR
high-frequency content, since gains also depend on how much detail there is to recover.

Writes <results>/coverage_<aug>.csv and .md, and a scatter plot <results>/coverage_<aug>.png
if matplotlib is available.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.bursts import discover, read_rgb  # noqa: E402


def phases(log: dict, num_frames: int) -> np.ndarray:
    keys = sorted(log)[:num_frames]
    d = np.array([[log[k]["dx_pixels_lr"], log[k]["dy_pixels_lr"]] for k in keys], dtype=np.float64)
    return np.mod(d, 1.0)


def coverage(ph: np.ndarray, k: int) -> float:
    cells = np.floor(ph * k).astype(int).clip(0, k - 1)
    return len({(int(a), int(b)) for a, b in cells}) / float(k * k)


def fill_distance(ph: np.ndarray, n: int = 64) -> float:
    """Largest distance from any point of the unit torus to its nearest phase (LR px)."""
    g = (np.arange(n) + 0.5) / n
    P = np.stack(np.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)
    diff = np.abs(P[:, None, :] - ph[None, :, :])
    diff = np.minimum(diff, 1.0 - diff)
    return float(np.sqrt((diff ** 2).sum(-1)).min(1).max())


def hf_fraction(hr: np.ndarray, df: int) -> float:
    """Share of the HR image's (mean-removed) spectral energy above the LR Nyquist frequency."""
    g = hr.mean(-1)
    g = g - g.mean()
    F = np.abs(np.fft.fft2(g)) ** 2
    fy = np.fft.fftfreq(g.shape[0])[:, None]
    fx = np.fft.fftfreq(g.shape[1])[None, :]
    hi = (np.abs(fx) > 0.5 / df) | (np.abs(fy) > 0.5 / df)
    return float(F[hi].sum() / max(F.sum(), 1e-12))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def partial_corr(x, y, z):
    """corr(x, y) after removing a linear effect of z from both."""
    x, y, z = (np.asarray(t, float) for t in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 4:
        return float("nan")
    Z = np.stack([np.ones(m.sum()), z[m]], 1)
    rx = x[m] - Z @ np.linalg.lstsq(Z, x[m], rcond=None)[0]
    ry = y[m] - Z @ np.linalg.lstsq(Z, y[m], rcond=None)[0]
    return corr(rx, ry)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/satsynthburst")
    ap.add_argument("--results", default="results")
    ap.add_argument("--aug", default="light")
    ap.add_argument("--shift", default="1.0")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--baseline", default="bicubic", help="gain = method PSNR - this method's PSNR")
    ap.add_argument("--methods", nargs="*", default=["superf_mse", "superf_gnll", "gsmisr", "s2superres", "nir"])
    a = ap.parse_args()
    res = Path(a.results)

    rows = []
    for ref in discover(a.data_root, shift=a.shift, aug=a.aug):
        log = json.loads((Path(ref.path) / "transform_log.json").read_text())
        ph = phases(log, a.num_frames)
        hr = read_rgb(Path(ref.path) / "hr_ground_truth.png")
        rows.append({"scene": ref.scene, "df": ref.df, "n": len(ph),
                     "cov_df": coverage(ph, ref.df), "cov2": coverage(ph, 2), "cov4": coverage(ph, 4),
                     "cov8": coverage(ph, 8), "fill": fill_distance(ph), "hf": hf_fraction(hr, ref.df)})
    if not rows:
        sys.exit(f"no bursts under {a.data_root} for aug={a.aug}")

    scores = None
    sp = res / "scores.csv"
    if sp.exists():
        import pandas as pd
        s = pd.read_csv(sp, dtype={"aug": str, "shift": str})
        scores = s[(s.protocol == "paper") & (s.aug == a.aug)]

    def psnr(method, scene, df):
        if scores is None:
            return float("nan")
        v = scores[(scores.method == method) & (scores.scene == scene) & (scores.df == df)].psnr
        return float(v.iloc[0]) if len(v) else float("nan")

    for r in rows:
        base = psnr(a.baseline, r["scene"], r["df"])
        for m in a.methods:
            r[f"gain_{m}"] = psnr(m, r["scene"], r["df"]) - base

    out_csv = res / f"coverage_{a.aug}.csv"
    res.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    lines = [f"# Sub-pixel coverage vs gain (`{a.aug}`, first {a.num_frames} frames, gain over {a.baseline})", "",
             "| x | bursts | coverage at x (mean) | random-shift expectation | largest hole (LR px) |",
             "|---|---|---|---|---|"]
    for df in sorted({r["df"] for r in rows}):
        rs = [r for r in rows if r["df"] == df]
        n = rs[0]["n"]
        lines.append(f"| {df} | {len(rs)} | {np.mean([r['cov_df'] for r in rs]):.2f} "
                     f"(min {np.min([r['cov_df'] for r in rs]):.2f}, max {np.max([r['cov_df'] for r in rs]):.2f}) | "
                     f"{1 - (1 - 1 / df ** 2) ** n:.2f} | {np.mean([r['fill'] for r in rs]):.3f} |")
    lines += ["", "Correlation with per-scene gain (Pearson; partial = after removing the HR detail measure):", "",
              "| method | x | scenes | corr(gain, coverage) | partial | corr(gain, HR detail) |", "|---|---|---|---|---|---|"]
    for m in a.methods:
        for df in sorted({r["df"] for r in rows}):
            rs = [r for r in rows if r["df"] == df and np.isfinite(r[f"gain_{m}"])]
            if len(rs) < 3:
                continue
            g = [r[f"gain_{m}"] for r in rs]
            c = [r["cov_df"] for r in rs]
            hfv = [r["hf"] for r in rs]
            lines.append(f"| {m} | {df} | {len(rs)} | {corr(g, c):+.2f} | {partial_corr(g, c, hfv):+.2f} | {corr(g, hfv):+.2f} |")
    lines += ["", "With 20 scenes a correlation needs to be above about 0.44 in size to be distinguishable "
              "from zero at p = 0.05. Coverage varies little between bursts at x2 (almost always full), so "
              "expect the signal at x4 and x8."]
    (res / f"coverage_{a.aug}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dfs = sorted({r["df"] for r in rows})
        fig, axs = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 3.4))
        axs = np.atleast_1d(axs)
        for ax, df in zip(axs, dfs):
            for m, mk in zip(a.methods, "osd^v"):
                rs = [r for r in rows if r["df"] == df and np.isfinite(r[f"gain_{m}"])]
                if rs:
                    ax.scatter([r["cov_df"] for r in rs], [r[f"gain_{m}"] for r in rs], s=18, marker=mk, label=m)
            ax.set_title(f"x{df}")
            ax.set_xlabel(f"sub-pixel coverage at x{df}")
            ax.set_ylabel(f"PSNR gain over {a.baseline} (dB)")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(res / f"coverage_{a.aug}.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print(f"(no plot: {e})")


if __name__ == "__main__":
    main()
