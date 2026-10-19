#!/usr/bin/env python3
"""Timing table: our method and SuperF on the CPU (the main-table runs) and on one H200 (the *_h200 runs), on the
same bursts, with the PSNR of both so a speed number is never quoted for a run that scores differently.

    python eval/timing.py --results results > results/timing.md      (also writes results/timing_bursts.csv)

Time per burst = the wall time each runner records in meta.json (runtime_s):
  ours (runners/run_gsmisr.py): the whole fit() call: registration search, warm-up, 600 iterations, final render
  SuperF (runners/run_superf.py): its 2000 training iterations (model set-up and the final render excluded, ~1 s)
  GSASR (runners/run_gsasr.py): one forward pass + render of frame 0
CPU runs: 8 threads per burst, 4 bursts at a time on a 32-core cpup node. H200 runs: one fit at a time on one GPU
(WORKERS=1). PSNR: paper protocol (scores.csv, written by slurm/report.sbatch).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PAIRS = [("ours (gsmisr_tv)", "gsmisr_tv", "gsmisr_tv_h200"),
         ("SuperF MSE", "superf_mse", "superf_mse_h200"),
         ("SuperF GNLL", "superf_gnll", "superf_gnll_h200")]
SINGLE = [("GSASR EDSR (frame 0 only)", "gsasr_edsr"), ("GSASR RDN (frame 0 only)", "gsasr_rdn")]


def meta_rows(results: Path, tag: str, aug: str) -> pd.DataFrame:
    rows = []
    for f in sorted((results / tag).glob(f"*/scale_*_aug_{aug}/meta.json")):
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        b = m.get("burst", {})
        rows.append({"tag": tag, "scene": b.get("scene", f.parent.parent.name), "df": float(b.get("df", "nan")),
                     "time_s": m.get("runtime_s"), "gpu_peak_mb": m.get("gpu_peak_mb"),
                     "device": m.get("device"), "gpu_name": m.get("gpu_name")})
    return pd.DataFrame(rows, columns=["tag", "scene", "df", "time_s", "gpu_peak_mb", "device", "gpu_name"])


def fmt(v, nd=1):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--aug", default="light")
    ap.add_argument("--scores", nargs="+", default=None,
                    help="score files to read (default <results>/scores.csv); later files win on duplicates")
    a = ap.parse_args()
    res = Path(a.results)
    files = [Path(f) for f in (a.scores or [res / "scores.csv"]) if Path(f).exists()]
    sc = (pd.concat([pd.read_csv(f, dtype={"shift": str, "aug": str}) for f in files], ignore_index=True)
          if files else pd.DataFrame())
    if not sc.empty:
        sc = sc.drop_duplicates(subset=["method", "scene", "df", "aug", "protocol"], keep="last")
        sc = sc[(sc.protocol == "paper") & (sc.aug == a.aug)][["method", "scene", "df", "psnr"]]
        sc["df"] = sc["df"].astype(float)

    def psnr(tag, keys):
        if sc.empty:
            return None
        s = sc[sc.method == tag].merge(keys, on=["scene", "df"])
        return s.psnr.mean() if len(s) == len(keys) and len(s) else None

    per_burst, lines = [], []
    gpu_names = set()
    lines += ["| Method | Scale | Bursts | CPU s/burst | H200 s/burst | Speed-up | H200 peak GB | PSNR CPU | PSNR H200 |",
              "|---|---|---|---|---|---|---|---|---|"]
    gpu_mean = {}
    for name, cpu_tag, gpu_tag in PAIRS:
        g = meta_rows(res, gpu_tag, a.aug)
        if g.empty:
            continue
        c = meta_rows(res, cpu_tag, a.aug)
        gpu_names |= set(g.gpu_name.dropna())
        for df in sorted(g.df.unique()):
            gg = g[g.df == df]
            keys = gg[["scene", "df"]]
            cc = c.merge(keys, on=["scene", "df"])
            both = gg.merge(cc, on=["scene", "df"], suffixes=("_gpu", "_cpu"))
            for _, r in both.iterrows():
                per_burst.append({"method": name, "scene": r.scene, "df": df, "cpu_s": r.time_s_cpu,
                                  "h200_s": r.time_s_gpu, "h200_peak_mb": r.gpu_peak_mb_gpu})
            t_gpu, t_cpu = gg.time_s.mean(), (cc.time_s.mean() if len(cc) == len(gg) else None)
            gpu_mean[(cpu_tag, df)] = (t_gpu, psnr(gpu_tag, keys), keys)
            lines.append(f"| {name} | x{df:g} | {len(gg)} | {fmt(t_cpu, 0)} | {fmt(t_gpu, 1)} | "
                         f"{fmt(None if t_cpu is None else t_cpu / t_gpu, 1)}x | "
                         f"{fmt(gg.gpu_peak_mb.max() / 1024 if gg.gpu_peak_mb.notna().any() else None, 2)} | "
                         f"{fmt(psnr(cpu_tag, keys), 2)} | {fmt(psnr(gpu_tag, keys), 2)} |")
    if len(lines) == 2:
        print("no *_h200 runs found under", res)
        return 1
    out = ["## Timing: CPU vs one H200, same bursts", "",
           f"GPU: {', '.join(sorted(gpu_names)) or 'unknown'}. CPU: 8 threads per burst (4 bursts at a time on a "
           "32-core node). Time = the optimisation wall time per burst from meta.json (see eval/timing.py).", ""]
    out += lines
    # ours vs SuperF on the same GPU, same bursts
    rows = ["", "### On the H200: ours vs SuperF, same bursts", "",
            "| Scale | ours s/burst | SuperF MSE s/burst | SuperF GNLL s/burst | ours faster than GNLL by | "
            "PSNR ours - GNLL |", "|---|---|---|---|---|---|"]
    for df in sorted({k[1] for k in gpu_mean}):
        o, m, gn = (gpu_mean.get((t, df)) for t in ("gsmisr_tv", "superf_mse", "superf_gnll"))
        if not o:
            continue
        rows.append(f"| x{df:g} | {fmt(o[0])} | {fmt(m[0] if m else None)} | {fmt(gn[0] if gn else None)} | "
                    f"{fmt(gn[0] / o[0] if gn else None, 1)}x | "
                    f"{fmt(o[1] - gn[1] if gn and o[1] is not None and gn[1] is not None else None, 2)} |")
    out += rows
    single = []
    for name, tag in SINGLE:
        s = meta_rows(res, tag, a.aug)
        for df in sorted(s.df.unique()) if not s.empty else []:
            ss = s[s.df == df]
            single.append(f"| {name} | x{df:g} | {len(ss)} | {fmt(ss.time_s.mean(), 2)} | "
                          f"{fmt(psnr(tag, ss[['scene', 'df']]), 2)} |")
    if single:
        out += ["", "### Single-image GSASR on the H200 (all bursts)", "",
                "| Method | Scale | Bursts | s/image | PSNR |", "|---|---|---|---|---|"] + single
    print("\n".join(out))
    pd.DataFrame(per_burst).to_csv(res / "timing_bursts.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
