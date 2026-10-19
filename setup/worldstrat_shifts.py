#!/usr/bin/env python3
"""How much sub-pixel shift diversity do WorldStrat's real Sentinel-2 revisits have?

Multi-frame SR needs frames that sample the ground at different sub-pixel offsets. WorldStrat's revisits are
delivered resampled onto one common map grid, so the only offsets left between them come from residual
geolocation error. This measures them before any SR run.

For every AOI, every revisit that is cloud-free and fully valid in the central crop is registered, on the mean of
the 10 m bands (B02, B03, B04, B08), to the clear revisit that best matches all the others (the medoid). Registration
is phase correlation with a Gaussian low-pass weight (aliasing sits in the high frequencies; error about 0.005 LR px
on synthetic box-averaged frames, against 0.06 px unweighted), refined on a 0.01 px grid by a matrix DFT. Every frame
is also registered to a second reference; the closure error s(i, ref) - s(i, ref2) - s(ref2, ref) is 0 for a
consistent frame, so its rms over the frames is the noise floor of the measurement. A frame is a registration
failure (snow, haze, scene change) if its shift exceeds --max_shift or its closure error exceeds --max_closure.

Per AOI it reports, over the clear frames that did not fail: the spread of the shifts, the ×4 sub-pixel phases they
occupy (each shift rounded to the nearest quarter pixel relative to the reference, so 16 possible phases; 16 random
shifts as in SatSynthBurst occupy 0.64 of them on average), the share of revisits at the reference's own ×4 phase,
and the noise floor.

    python3 setup/worldstrat_shifts.py --root data/worldstrat/aoi --aois setup/worldstrat_aois.txt \
        --out data/worldstrat/shift_diversity

Needs numpy and tifffile. Writes <out>_frames.csv and <out>_aois.csv and prints the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

BANDS_10M = [1, 2, 3, 7]          # B02, B03, B04, B08 in WorldStrat's 12-band L2A order (B01, B02, ..., B12)


def read_groups(path):
    """AOI -> group from the comment headers of worldstrat_aois.txt (sweet / bitter / satsynthburst)."""
    groups, cur = {}, "other"
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            low = s.lower()
            if "sweet" in low and "bitter" not in low:
                cur = "sweet"
            elif "bitter" in low and "sweet" not in low:
                cur = "bitter"
            elif "satsynthburst" in low:
                cur = "satsynthburst"
            continue
        name = s.split("#", 1)[0].strip()
        if name:
            groups.setdefault(name, cur)    # first listing wins (one AOI is in both sweet and SatSynthBurst)
    return groups


def register(ref, mov, up=100, f0=0.15):
    """Shift (dy, dx) in pixels such that mov(y, x) ~ ref(y - dy, x - dx), and the peak height (1 = perfect match).
    Phase correlation weighted by exp(-|f|^2 / f0^2) (f in cycles/px), refined on an up x finer grid around the
    integer peak by a matrix DFT."""
    ny, nx = ref.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    a = (ref - ref.mean()) * win
    b = (mov - mov.mean()) * win
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    wgt = np.exp(-(fy[:, None] ** 2 + fx[None, :] ** 2) / f0 ** 2)
    R = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    R = R / (np.abs(R) + 1e-12) * wgt
    cc = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    py = py - ny if py > ny // 2 else py
    px = px - nx if px > nx // 2 else px
    k = np.arange(-1.5, 1.5 + 1e-9, 1.0 / up)
    ry, rx = py + k, px + k
    ey = np.exp(2j * np.pi * np.outer(ry, fy))          # rows: candidate y shifts
    ex = np.exp(2j * np.pi * np.outer(fx, rx))          # cols: candidate x shifts
    up_cc = (ey @ R @ ex).real / wgt.sum()
    iy, ix = np.unravel_index(np.argmax(up_cc), up_cc.shape)
    return float(ry[iy]), float(rx[ix]), float(up_cc[iy, ix])


def medoid_order(imgs, ids, f0=0.15):
    """Frames ordered by their median match (integer-peak height of the weighted phase correlation) to the other
    frames, best first. The best one is the reference: a frame with snow, haze or a large scene change matches
    nothing and would make every shift fail."""
    if len(ids) < 3:
        return list(ids)
    ny, nx = imgs[ids[0]].shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    fy, fx = np.fft.fftfreq(ny), np.fft.fftfreq(nx)
    wgt = np.exp(-(fy[:, None] ** 2 + fx[None, :] ** 2) / f0 ** 2)
    F = {i: np.fft.fft2((imgs[i] - imgs[i].mean()) * win) for i in ids}
    score = {i: [] for i in ids}
    for a_, i in enumerate(ids):
        for j in ids[a_ + 1:]:
            R = F[j] * np.conj(F[i])
            v = float(np.fft.ifft2(R / (np.abs(R) + 1e-12) * wgt).real.max() / wgt.sum())
            score[i].append(v)
            score[j].append(v)
    return sorted(ids, key=lambda i: -float(np.median(score[i])))


def frames_of(aoi_dir: Path, name: str):
    out = []
    for p in (aoi_dir / "L2A").glob(f"{name}-*-L2A_data.tiff"):
        idx = int(p.name[len(name) + 1:].split("-", 1)[0])
        out.append((idx, p))
    return sorted(out)


def delta_days(m):
    """|LR - HR date| in days; WorldStrat stores it as an int (Landcover, UNHCR) or a string (Amnesty, ASMSpotter)."""
    try:
        return abs(float(m.get("delta", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def mask_mean(p: Path, sl, scale=1.0, thresh=None):
    if not p.exists():
        return float("nan")
    m = tifffile.imread(p)[sl].astype(np.float64) / scale
    return float((m > thresh).mean() if thresh is not None else m.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/worldstrat/aoi")
    ap.add_argument("--aois", default="setup/worldstrat_aois.txt")
    ap.add_argument("--out", default="data/worldstrat/shift_diversity")
    ap.add_argument("--crop", type=int, default=128, help="central crop (LR px) used for registration")
    ap.add_argument("--max_cloud", type=float, default=0.01, help="max cloud-mask (CLM) fraction in the crop")
    ap.add_argument("--max_clp", type=float, default=0.05,
                    help="max fraction of crop pixels with cloud probability (CLP) above 0.4")
    ap.add_argument("--f0", type=float, default=0.15, help="low-pass cut-off of the phase correlation (cycles/px)")
    ap.add_argument("--max_shift", type=float, default=3.0,
                    help="shifts above this (LR px) are registration failures (snow, haze, scene change)")
    ap.add_argument("--max_closure", type=float, default=0.25,
                    help="frames whose two registrations disagree by more than this (LR px) are failures too")
    ap.add_argument("--only", default="", help="comma-separated AOI names (default: all in --aois)")
    a = ap.parse_args()

    groups = read_groups(a.aois)
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        groups = {k: v for k, v in groups.items() if k in keep}
    root = Path(a.root)
    frows, arows = [], []
    for name, group in groups.items():
        d = root / name
        fr = frames_of(d, name)
        if not fr:
            print(f"!!! {name}: no L2A frames under {d}")
            continue
        imgs, cloud, clp, valid, meta = {}, {}, {}, {}, {}
        for idx, p in fr:
            x = tifffile.imread(p).astype(np.float64)
            H, W = x.shape[:2]
            c = min(a.crop, H, W)
            y0, x0 = (H - c) // 2, (W - c) // 2
            sl = (slice(y0, y0 + c), slice(x0, x0 + c))
            imgs[idx] = x[sl][..., BANDS_10M].mean(-1)
            stem = p.name[: -len("L2A_data.tiff")]
            cloud[idx] = mask_mean(p.with_name(stem + "CLM.tiff"), sl)
            clp[idx] = mask_mean(p.with_name(stem + "CLP.tiff"), sl, scale=255.0, thresh=0.4)
            dm = mask_mean(p.with_name(stem + "dataMask.tiff"), sl)
            valid[idx] = 1.0 if dm != dm else dm
            if not np.isfinite(imgs[idx]).all():
                valid[idx] = 0.0
            try:
                meta[idx] = json.loads(p.with_name(stem[:-1] + ".metadata").read_text())
            except Exception:                     # noqa: BLE001
                meta[idx] = {}
        ok = [i for i in imgs if valid[i] > 0.999 and not (cloud[i] > a.max_cloud) and not (clp[i] > a.max_clp)]
        if not ok:
            print(f"!!! {name}: no clear revisit")
            arows.append({"aoi": name, "group": group, "n_frames": len(imgs), "n_clear": 0})
            continue
        order = medoid_order(imgs, sorted(ok, key=lambda i: (cloud[i], clp[i], delta_days(meta[i]), i)), f0=a.f0)
        ref = order[0]
        reg = {i: (0.0, 0.0, 1.0) if i == ref else register(imgs[ref], imgs[i], f0=a.f0) for i in sorted(imgs)}
        far = {i for i in ok if max(abs(reg[i][0]), abs(reg[i][1])) > a.max_shift}
        good = [i for i in order if i not in far]
        ref2 = good[1] if len(good) > 2 else None
        clo = {}                                    # closure error per frame (LR px); 0 for a consistent frame
        if ref2 is not None:
            for i in good[2:]:
                sy, sx, _ = register(imgs[ref2], imgs[i], f0=a.f0)
                clo[i] = (reg[i][0] - sy - reg[ref2][0], reg[i][1] - sx - reg[ref2][1])
        failed = far | {i for i, c in clo.items() if max(abs(c[0]), abs(c[1])) > a.max_closure}
        used = [i for i in sorted(ok) if i not in failed]
        for i in sorted(imgs):
            dy, dx, peak = reg[i]
            m = meta[i]
            c = clo.get(i)
            frows.append({"aoi": name, "group": group, "frame": i, "ref": int(i == ref), "ref2": int(i == ref2),
                          "clear": int(i in ok), "failed": int(i in failed),
                          "cloud": round(cloud[i], 4), "clp_high": round(clp[i], 4), "valid": round(valid[i], 4),
                          "lowres_date": m.get("lowres_date", m.get("datetime", "")), "delta_days": m.get("delta", ""),
                          "dy": round(dy, 3), "dx": round(dx, 3), "peak": round(peak, 3),
                          "closure_dy": round(c[0], 3) if c else "", "closure_dx": round(c[1], 3) if c else ""})
        if not used:
            arows.append({"aoi": name, "group": group, "n_frames": len(imgs), "n_clear": len(ok),
                          "n_failed": len(failed), "n_used": 0})
            continue
        s = np.array([(reg[i][0], reg[i][1], reg[i][2], int(i == ref)) for i in used])
        closure = [v for i in used if i in clo for v in clo[i]]
        n = len(s)
        others = s[s[:, 3] == 0]
        cells = {(int(np.round(4 * u)) % 4, int(np.round(4 * v)) % 4) for u, v in s[:, :2]}
        same = [(int(np.round(4 * u)) % 4, int(np.round(4 * v)) % 4) == (0, 0) for u, v in others[:, :2]]
        arows.append({
            "aoi": name, "group": group, "n_frames": len(imgs), "n_clear": len(ok), "n_failed": len(failed),
            "n_used": n,
            "std_dy": round(float(s[:, 0].std()), 3), "std_dx": round(float(s[:, 1].std()), 3),
            "max_abs_shift": round(float(np.abs(s[:, :2]).max()), 3),
            "cover_x4": round(len(cells) / 16, 3),
            "cover_x4_random": round(1 - (1 - 1 / 16) ** n, 3),
            "same_phase_x4": round(float(np.mean(same)), 3) if same else "",
            "closure_rms": round(float(np.sqrt(np.mean(np.square(closure)))), 3) if closure else "",
            "median_peak": round(float(np.median(others[:, 2])), 3) if len(others) else "",
        })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for rows, suffix in ((frows, "_frames.csv"), (arows, "_aois.csv")):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(f"{out}{suffix}", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else f"{'-':>{spec.split('.')[0] or 1}}"

    print(f"{'AOI':28s} {'group':13s} clear fail used std_dy std_dx max|s|  cover_x4 (random) same_x4 closure  peak")
    for r in arows:
        if r.get("n_used"):
            print(f"{r['aoi'][:28]:28s} {r['group']:13s} {r['n_clear']:2d}/{r['n_frames']:2d} {r['n_failed']:3d} "
                  f"{r['n_used']:4d} {r['std_dy']:6.3f} "
                  f"{r['std_dx']:6.3f} {r['max_abs_shift']:6.3f}    {r['cover_x4']:.2f}  ({r['cover_x4_random']:.2f})"
                  f"   {fmt(r['same_phase_x4'], '5.2f')}   {fmt(r['closure_rms'], '5.3f')}  {fmt(r['median_peak'], '5.2f')}")
        else:
            print(f"{r['aoi'][:28]:28s} {r['group']:13s} no usable revisit")
    for g in ("sweet", "bitter", "satsynthburst", "all"):
        rs = [r for r in arows if (g == "all" or r["group"] == g) and r.get("n_used", 0) > 2]
        if rs:
            def med(k):
                v = [r[k] for r in rs if isinstance(r[k], (int, float))]
                return float(np.median(v)) if v else float("nan")
            print(f"== {g:13s} AOIs {len(rs):2d} | used frames {med('n_used'):.0f} | std (dy, dx) "
                  f"({med('std_dy'):.3f}, {med('std_dx'):.3f}) LR px | cover_x4 {med('cover_x4'):.2f} vs "
                  f"{med('cover_x4_random'):.2f} random | same x4 phase as ref {med('same_phase_x4'):.2f} | "
                  f"noise floor (closure rms) {med('closure_rms'):.3f} px | failed frames "
                  f"{sum(r['n_failed'] for r in rs)}")


if __name__ == "__main__":
    main()
