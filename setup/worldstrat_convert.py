#!/usr/bin/env python3
"""WorldStrat AOIs -> bursts in our layout (common/bursts.py), so every runner and the scorer work unchanged.

    python3 setup/worldstrat_convert.py --root data/worldstrat/aoi --aois setup/worldstrat_aois.txt \
        --shifts data/worldstrat/shift_diversity_frames.csv --out data/worldstrat_bursts

Protocol (x4, the geometry of SuperF's sweet/bitter loader, WorldStratTestDataset):
  * LR: Sentinel-2 L2A B04/B03/B02 (true colour), central 64x64 crop of the revisit raster (all revisits share
    one grid). One linear gain per AOI (1.25 x the 99.9th percentile of the reference crop -> 255), 8-bit PNG.
  * HR: SPOT pansharpened RGB (_ps.tiff, linear DN; same bounding box as the LR raster), INTER_AREA-resized to
    4x the LR raster size, as SuperF does. Then, unlike SuperF, registered to the reference revisit: the shift of
    every clear revisit relative to the downsampled HR (phase correlation, mean of R/G/B/NIR vs mean of
    B02/B03/B04/B08) minus its shift relative to the reference gives the HR-to-reference offset; the median is
    taken, first on the central 128x128 LR px (robust), then refined on the central 80x80 (the scored region; the
    refinement is kept if it moves the offset by <= 0.5 px). The 256x256 crop is resampled (bicubic) at that
    offset, so hr_ground_truth.png is aligned with
    sample_00 the way SatSynthBurst's is and the paper protocol can score it as is. One gain per AOI
    (1.05 x the 99.9th percentile of the crop -> 255), 8-bit PNG.
  * Frames: sample_00 = the clear revisit closest in date to the HR image; then the other clear revisits by
    |date gap|; then cloudy revisits and registration failures (from --shifts). All 16 are written, so
    --num_frames N in the runners takes the N best. Clear = cloud mask <= 1 % and cloud probability > 0.4 on
    <= 5 % of the LR crop.
  * transform_log.json: SuperF's loader needs the six shift keys; they are 0.0 placeholders with
    "shifts_known": false (common/bursts.true_shifts_lr then returns None). The estimated shifts relative to
    sample_00 are stored as est_dx_lr / est_dy_lr (phase correlation on the central 128x128), for analysis only,
    in the generator's sign convention (output(x) = input(x + d), the sign of dx_pixels_lr and of our method's
    final_shifts_lr_px).
  * HR check: the registered crop, downsampled, is registered again against sample_00's crop. If it still moves by
    more than 0.5 LR px, or the per-revisit offset estimates disagree (MAD > 0.25 px), the HR image does not match
    the revisits (scene change over a long date gap, repetitive fields): hr_registration_ok = false, and the AOI
    is listed under "hr_unreliable" in groups.json so the main table can leave it out.
  * Extra files per burst: clp_XX.png (cloud probability 0-255 on the LR crop, for per-pixel weights later) and
    burst_info.json (sources, offsets, gains, checks). <out>/groups.json lists the sweet / bitter / satsynthburst
    AOIs and the hr_unreliable ones; <out>/convert_summary.csv has one row per AOI.

Needs numpy, opencv and tifffile.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldstrat_shifts import delta_days, frames_of, register  # noqa: E402

RGB = [3, 2, 1]                 # B04, B03, B02 in WorldStrat's 12-band L2A order (B01, B02, ..., B12)
BANDS_10M = [1, 2, 3, 7]        # B02, B03, B04, B08: their mean is registered against the mean of SPOT R/G/B/NIR
VERSION = "worldstrat_convert v1.2 (18 Oct 2026)"
MAX_RESIDUAL = 0.5              # LR px: an HR crop that still moves this much against sample_00 is not a usable GT
MAX_MAD = 0.25                  # LR px: spread of the per-revisit HR offset estimates


def centre(h, w, c):
    return (h - c) // 2, (w - c) // 2


def all_groups(path):
    """AOI -> every group it is listed under, and group -> AOIs, from the comment headers of worldstrat_aois.txt
    (one AOI, UNHCR-SYRs008164, is in both sweet and SatSynthBurst)."""
    of, members, cur = {}, {}, "other"
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
            of.setdefault(name, [])
            if cur not in of[name]:
                of[name].append(cur)
                members.setdefault(cur, []).append(name)
    return of, members


def load_aoi(root: Path, name: str, crop: int):
    out = []
    for idx, p in frames_of(root / name, name):
        x = tifffile.imread(p).astype(np.float64)
        H, W = x.shape[:2]
        y0, x0 = centre(H, W, crop)
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        stem = p.name[: -len("L2A_data.tiff")]

        def mask(kind):
            q = p.with_name(stem + kind + ".tiff")
            return tifffile.imread(q).astype(np.float64) if q.exists() else None

        clm, clp, dm = mask("CLM"), mask("CLP"), mask("dataMask")
        try:
            meta = json.loads(p.with_name(stem[:-1] + ".metadata").read_text())
        except Exception:                     # noqa: BLE001
            meta = {}
        finite = bool(np.isfinite(x[sl]).all())
        out.append({
            "idx": idx, "file": p.name, "img": np.nan_to_num(x), "H": H, "W": W, "y0": y0, "x0": x0,
            "cloud": float(clm[sl].mean()) if clm is not None else float("nan"),
            "clp_high": float((clp[sl] / 255.0 > 0.4).mean()) if clp is not None else float("nan"),
            "valid": (float(dm[sl].mean()) if dm is not None else 1.0) if finite else 0.0,
            "clp_crop": clp[sl] if clp is not None else np.zeros((crop, crop)),
            "meta": meta,
        })
    return out


def lum10(x, sl):
    return x[sl][..., BANDS_10M].mean(-1)


def to8(x, white):
    return np.clip(np.round(x / white * 255.0), 0, 255).astype(np.uint8)


def write_rgb(path, rgb8):
    cv2.imwrite(str(path), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/worldstrat/aoi")
    ap.add_argument("--aois", default="setup/worldstrat_aois.txt")
    ap.add_argument("--shifts", default="data/worldstrat/shift_diversity_frames.csv",
                    help="per-frame CSV of setup/worldstrat_shifts.py (registration failures); optional")
    ap.add_argument("--out", default="data/worldstrat_bursts")
    ap.add_argument("--df", type=int, default=4)
    ap.add_argument("--lr_crop", type=int, default=64)
    ap.add_argument("--reg_crop", type=int, default=128, help="central crop (LR px) used for registration")
    ap.add_argument("--fine_crop", type=int, default=80, help="central crop (LR px) for refining the HR offset")
    ap.add_argument("--max_cloud", type=float, default=0.01)
    ap.add_argument("--max_clp", type=float, default=0.05)
    ap.add_argument("--max_shift", type=float, default=3.0, help="frame shifts above this (LR px) = failures")
    ap.add_argument("--only", default="", help="comma-separated AOI names (default: all in --aois)")
    a = ap.parse_args()

    df, c = a.df, a.lr_crop
    hc = c * df
    groups_of, members = all_groups(a.aois)
    names = list(groups_of)
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        names = [n for n in names if n in keep]

    failed_csv = set()
    if a.shifts and Path(a.shifts).exists():
        for r in csv.DictReader(open(a.shifts)):
            if r.get("failed") == "1":
                failed_csv.add((r["aoi"], int(r["frame"])))

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    root = Path(a.root)
    for name in names:
        fr = load_aoi(root, name, c)
        if not fr:
            print(f"!!! {name}: no L2A frames")
            continue
        H, W = fr[0]["H"], fr[0]["W"]
        if any((f["H"], f["W"]) != (H, W) for f in fr):
            print(f"!!! {name}: revisits on different grids, skipped")
            continue
        y0, x0 = fr[0]["y0"], fr[0]["x0"]
        ry0, rx0 = centre(H, W, a.reg_crop)
        rsl = (slice(ry0, ry0 + a.reg_crop), slice(rx0, rx0 + a.reg_crop))
        for f in fr:
            f["clear"] = f["valid"] > 0.999 and not (f["cloud"] > a.max_cloud) and not (f["clp_high"] > a.max_clp)
            f["gap"] = delta_days(f["meta"])
        cand = [f for f in fr if f["clear"] and (name, f["idx"]) not in failed_csv]
        if not cand:
            print(f"!!! {name}: no clear revisit in the {c}x{c} crop, skipped")
            summary.append({"aoi": name, "groups": "+".join(groups_of.get(name, [])), "status": "no clear revisit"})
            continue
        ref = min(cand, key=lambda f: (f["gap"], f["cloud"], f["idx"]))
        lref = lum10(ref["img"], rsl)
        for f in fr:
            f["s"] = (0.0, 0.0, 1.0) if f is ref else register(lref, lum10(f["img"], rsl))
            f["failed"] = (name, f["idx"]) in failed_csv or max(abs(f["s"][0]), abs(f["s"][1])) > a.max_shift
        good = [f for f in fr if f["clear"] and not f["failed"]]
        rest = [f for f in fr if not (f["clear"] and not f["failed"])]
        order = [ref] + sorted([f for f in good if f is not ref], key=lambda f: (f["gap"], f["idx"])) + \
            sorted(rest, key=lambda f: (f["failed"], f["cloud"] + f["clp_high"], f["gap"], f["idx"]))

        # HR: SPOT pansharpened R, G, B, NIR on 4x the LR raster grid (SuperF's resize), then registration
        hp = root / name / f"{name}_ps.tiff"
        hr = tifffile.imread(hp).astype(np.float32)
        hr4 = cv2.resize(hr, (W * df, H * df), interpolation=cv2.INTER_AREA)
        hrds = hr4.reshape(H, df, W, df, hr4.shape[2]).mean((1, 3)).astype(np.float64)
        lhr = hrds[rsl].mean(-1)
        est = []
        for f in good:
            sy, sx, pk = register(lhr, lum10(f["img"], rsl))
            est.append((sy - f["s"][0], sx - f["s"][1], pk))
        e = np.array(est)
        oy, ox = float(np.median(e[:, 0])), float(np.median(e[:, 1]))
        mad = float(np.median(np.abs(e[:, :2] - [oy, ox]))) if len(e) > 1 else float("nan")
        coarse = (oy, ox)
        fy0, fx0 = centre(H, W, a.fine_crop)       # refine on the scored region
        fsl = (slice(fy0, fy0 + a.fine_crop), slice(fx0, fx0 + a.fine_crop))
        lhr_f = hrds[fsl].mean(-1)
        e2 = np.array([(sy - f["s"][0], sx - f["s"][1]) for f in good
                       for sy, sx, _ in [register(lhr_f, lum10(f["img"], fsl))]])
        fy, fx = float(np.median(e2[:, 0])), float(np.median(e2[:, 1]))
        refined = abs(fy - oy) <= 0.5 and abs(fx - ox) <= 0.5
        if refined:
            oy, ox = fy, fx
        M = np.float32([[1, 0, df * (x0 - ox)], [0, 1, df * (y0 - oy)]])
        crop_rgb = cv2.warpAffine(hr4[..., :3], M, (hc, hc), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT)
        crop_rgb = np.maximum(crop_rgb, 0)
        crop4 = cv2.warpAffine(hr4, M, (hc, hc), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)
        down = crop4.reshape(c, df, c, df, crop4.shape[2]).mean((1, 3)).mean(-1).astype(np.float64)
        lsl = (slice(y0, y0 + c), slice(x0, x0 + c))
        res_y, res_x, res_pk = register(down, ref["img"][lsl][..., BANDS_10M].mean(-1))
        hr_ok = max(abs(res_y), abs(res_x)) <= MAX_RESIDUAL and not (mad > MAX_MAD)

        # write the burst
        d = out_root / name / f"scale_{df}_shift_0.0px_aug_real"
        d.mkdir(parents=True, exist_ok=True)
        lr_white = 1.25 * float(np.percentile(ref["img"][lsl][..., RGB], 99.9))
        hr_white = 1.05 * float(np.percentile(crop_rgb, 99.9))
        write_rgb(d / "hr_ground_truth.png", to8(crop_rgb, hr_white))
        log = {}
        for k, f in enumerate(order):
            key = f"sample_{k:02d}"
            write_rgb(d / f"{key}.png", to8(f["img"][lsl][..., RGB], lr_white))
            cv2.imwrite(str(d / f"clp_{k:02d}.png"), np.clip(f["clp_crop"], 0, 255).astype(np.uint8))
            m = f["meta"]
            log[key] = {
                "path": f"{key}.png", "shape": [c, c, 3], "augmentation": "real",
                "dx_pixels_lr": 0.0, "dy_pixels_lr": 0.0, "dx_pixels_hr": 0.0, "dy_pixels_hr": 0.0,
                "dx_percent": 0.0, "dy_percent": 0.0, "shifts_known": False,
                "est_dx_lr": round(-f["s"][1], 3), "est_dy_lr": round(-f["s"][0], 3), "est_peak": round(f["s"][2], 3),
                "source": f["file"], "revisit": f["idx"],
                "lowres_date": m.get("lowres_date", m.get("datetime", "")), "delta_days": f["gap"],
                "cloud_frac": round(f["cloud"], 4), "clp_high_frac": round(f["clp_high"], 4),
                "clear": bool(f["clear"]), "registration_failed": bool(f["failed"]),
                "downsampling": {"method": "real Sentinel-2 L2A revisit (WorldStrat)", "factor": df},
            }
        (d / "transform_log.json").write_text(json.dumps(log, indent=2))
        info = {
            "aoi": name, "groups": groups_of.get(name, []), "converter": VERSION, "df": df,
            "lr_raster": [H, W], "lr_crop_yx": [y0, x0, c], "reference_revisit": ref["idx"],
            "frame_order_revisits": [f["idx"] for f in order], "n_clear": len(good),
            "hr_source": hp.name, "hr_grid": [H * df, W * df],
            "hr_offset_lr_px": {"dy": round(oy, 3), "dx": round(ox, 3), "mad": round(mad, 3), "n": len(e),
                                "median_peak": round(float(np.median(e[:, 2])), 3),
                                "coarse_dy": round(coarse[0], 3), "coarse_dx": round(coarse[1], 3),
                                "refined": bool(refined)},
            "residual_after_registration_lr_px": {"dy": round(res_y, 3), "dx": round(res_x, 3), "peak": round(res_pk, 3)},
            "hr_registration_ok": bool(hr_ok),
            "lr_white_reflectance": lr_white, "hr_white_dn": hr_white,
            "bands": {"lr": "B04, B03, B02", "hr": "SPOT pansharpened R, G, B", "registration": "mean B02/B03/B04/B08 vs mean R/G/B/NIR"},
        }
        (d / "burst_info.json").write_text(json.dumps(info, indent=2))
        summary.append({"aoi": name, "groups": "+".join(groups_of.get(name, [])), "status": "ok",
                        "ref_revisit": ref["idx"], "ref_gap_days": ref["gap"], "n_clear": len(good),
                        "n_failed": sum(f["failed"] for f in fr),
                        "hr_dy": round(oy, 3), "hr_dx": round(ox, 3), "hr_mad": round(mad, 3),
                        "hr_peak": round(float(np.median(e[:, 2])), 3),
                        "res_dy": round(res_y, 3), "res_dx": round(res_x, 3), "hr_ok": int(hr_ok)})
        s = summary[-1]
        print(f"{name[:28]:28s} {s['groups']:20s} ref r{s['ref_revisit']:<2d} gap {s['ref_gap_days']:5.0f} d  "
              f"clear {s['n_clear']:2d}  HR offset ({oy:+.2f}, {ox:+.2f}) mad {mad:.2f} peak {s['hr_peak']:.2f}  "
              f"residual ({res_y:+.2f}, {res_x:+.2f})" + ("" if hr_ok else "  << HR unreliable"), flush=True)
    keys = list(dict.fromkeys(k for r in summary for k in r))
    with open(out_root / "convert_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    ok = [r for r in summary if r["status"] == "ok"]
    bad = [r["aoi"] for r in ok if not r["hr_ok"]]
    if not a.only:
        (out_root / "groups.json").write_text(json.dumps({**members, "hr_unreliable": bad}, indent=2))
    print(f"\n{len(ok)} bursts written to {out_root}, {len(summary) - len(ok)} AOIs skipped; "
          f"HR unreliable on {len(bad)}: {', '.join(bad) if bad else '-'}")


if __name__ == "__main__":
    main()
