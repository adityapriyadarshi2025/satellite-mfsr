#!/usr/bin/env python3
"""s2-superres (Robinson & Corley, blog + MIT code, May 2026) on SatSynthBurst.
The one runnable multi-frame Gaussian-splat SR method, so it is the Gaussian row.

    python runners/run_s2superres.py --s2_dir baselines/s2-superres --device cuda:0
    python runners/run_s2superres.py --as_released --tag s2superres_asis   # the released code, no fixes

The released solver is written for real Sentinel-2 stacks. Three things stop it
scoring fairly against a ground-truth HR image; each fix is switchable:

  1. shift sign (--no_fix_sign): their phase correlation returns minus the content
     shift, while SplatField.predict_lr treats +shift as content moving +x/+y
     (LR pixel j integrates [j - s, j + 1 - s]). Checked numerically: a +1.3 px
     content shift comes back as about -0.94. So the starting shifts are negated.
  2. HR render (--no_fix_render): render_hr() bicubic-resizes the weight grid, and
     splat k sits at k/S (the left edge of HR pixel k), so the output is off by half
     an HR pixel. The fix renders each HR pixel as the analytic integral of the
     splat field over that HR pixel's footprint (same erf maths, no sensor PSF).
  3. frame edges (--margin 0): splats past the grid edge count as zeros, so shifted
     frames get dark borders. The fix pads the splat grid by --margin LR px
     (replicate-initialised) and crops back.

PSF: SatSynthBurst blurs with sigma = 1/df HR px = 1/df^2 LR px before box
averaging, so --psf auto uses that. Their Sentinel-2 default is 0.5 LR px.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.bursts import add_common_args, select, load_burst, save_prediction, pred_dir, true_shifts_lr  # noqa: E402


def build_args():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--s2_dir", default=os.path.join(os.environ.get("BASELINES", "baselines"), "s2-superres"))
    p.add_argument("--tag", default="s2superres")
    p.add_argument("--device", default="auto")
    p.add_argument("--strategy", default="lbfgs_adam", choices=["adam", "lbfgs", "adam_lbfgs", "lbfgs_adam"])
    p.add_argument("--psf", default="auto", help="sensor PSF sigma in LR px, or 'auto' = 1/df^2")
    p.add_argument("--sigma_splat", type=float, default=0.2, help="splat width in LR px (their default)")
    p.add_argument("--c2f", default="auto", help="coarse-to-fine splat scales ending at df, e.g. 2,4")
    p.add_argument("--tv", type=float, default=1e-3)
    p.add_argument("--huber_delta", type=float, default=0.003)
    p.add_argument("--n_iters", type=int, default=2000, help="adam strategy only")
    p.add_argument("--lbfgs_outer", type=int, default=8)
    p.add_argument("--adam_warmup", type=int, default=100)
    p.add_argument("--adam_anneal", type=int, default=500)
    p.add_argument("--adam_unreg", type=int, default=1000)
    p.add_argument("--lr_weights", type=float, default=2e-3)
    p.add_argument("--lr_shifts", type=float, default=1e-2)
    p.add_argument("--margin", type=int, default=4, help="LR px of extra splat grid on each side (0 = as released)")
    p.add_argument("--no_fix_sign", action="store_true")
    p.add_argument("--no_fix_render", action="store_true")
    p.add_argument("--as_released", action="store_true", help="all three fixes off")
    p.add_argument("--init", choices=["phasecorr", "search"], default="phasecorr",
                   help="starting shifts: their phase correlation (as released, sign-fixed) or our exhaustive "
                        "NCC search (gsmisr.misr.register_search), to test how much registration explains")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.as_released:
        a.no_fix_sign, a.no_fix_render, a.margin = True, True, 0
    return a


def make_classes(SplatField):
    import torch
    import torch.nn.functional as F

    class MarginSplatField(SplatField):
        """SplatField on a grid padded by `margin` LR px per side; predictions cropped back."""

        def __init__(self, lr_h, lr_w, n_channels, splat_scale=4, sigma_splat=0.2, sigma_psf=0.5, margin=0):
            super().__init__(lr_h + 2 * margin, lr_w + 2 * margin, n_channels,
                             splat_scale=splat_scale, sigma_splat=sigma_splat, sigma_psf=sigma_psf)
            self.margin, self.inner_h, self.inner_w = margin, lr_h, lr_w

        def predict_lr(self, shifts):
            m = self.margin
            return super().predict_lr(shifts)[:, :, m:m + self.inner_h, m:m + self.inner_w]

        def init_from_observations(self, observations):
            m = self.margin
            obs = F.pad(observations, (m, m, m, m), mode="replicate") if m else observations
            super().init_from_observations(obs)

        def render_hr_exact(self, df):
            """HR pixel p covers [p/df, (p+1)/df] in LR units; splat k sits at k/S - margin.
            Value = Gaussian-weighted average of splat weights over that footprint (rows normalised
            to 1, like predict_lr), sigma = sigma_splat (the sensor PSF is not part of the HR image)."""
            S, m = self.S, self.margin
            dev = self.weights.device
            denom = self.sigma_splat * math.sqrt(2.0)

            def axis_matrix(n_hr, n_splat):
                p = torch.arange(n_hr, dtype=torch.float64, device=dev)[:, None]
                k = torch.arange(n_splat, dtype=torch.float64, device=dev)[None, :] / S - m
                K = 0.5 * (torch.erf(((p + 1) / df - k) / denom) - torch.erf((p / df - k) / denom))
                return (K / K.sum(dim=1, keepdim=True).clamp(min=1e-12)).float()

            Ky = axis_matrix(self.inner_h * df, self.splat_h)
            Kx = axis_matrix(self.inner_w * df, self.splat_w)
            return torch.einsum("hi,cij,wj->chw", Ky, self.weights, Kx)

        def render_hr_as_released(self, df):
            m, S = self.margin, self.S
            w = self.weights[:, m * S: m * S + self.inner_h * S, m * S: m * S + self.inner_w * S]
            return F.interpolate(w[None], size=(self.inner_h * df, self.inner_w * df),
                                 mode="bicubic", align_corners=False)[0].clamp(0, 1.5)

    return MarginSplatField


def main():
    args = build_args()
    s2 = Path(args.s2_dir).resolve()
    if not (s2 / "src" / "splat_field.py").exists():
        raise SystemExit(f"s2-superres not found at {s2}. Run slurm/setup.sh (repos step)")
    sys.path.insert(0, str(s2))

    import torch
    import src.strategies as strategies
    from src.splat_field import SplatField
    from src.shift_estimation import estimate_shifts_phase_correlation
    MarginSplatField = make_classes(SplatField)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    for ref in select(args):
        if args.skip_existing and (pred_dir(args.results, args.tag, ref) / "pred.npy").exists():
            continue
        torch.manual_seed(args.seed)
        df = ref.df
        psf = 1.0 / df ** 2 if args.psf == "auto" else float(args.psf)
        c2f = tuple(int(s) for s in args.c2f.split(",")) if args.c2f != "auto" else \
            {2: (2,), 4: (2, 4), 8: (2, 4, 8)}.get(df, (df,))
        if c2f[-1] != df:
            raise SystemExit(f"--c2f must end at df={df}, got {c2f}")

        frames, hr, log = load_burst(ref, args.num_frames)
        obs = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()  # [T,C,h,w]
        T, C, h, w = obs.shape

        if args.init == "search":
            from gsmisr.misr import register_search   # same convention as predict_lr: +b = content moves +b
            init = torch.from_numpy(register_search(frames)).float()
        else:
            init = estimate_shifts_phase_correlation(obs.clone())
            if not args.no_fix_sign:
                init = -init
        shifts = init.clone().to(device).requires_grad_(True)
        obs = obs.to(device)

        margin = args.margin
        strategies._make_field = lambda lr_h, lr_w, n_channels, splat_scale, sigma_splat, sigma_psf, dev: \
            MarginSplatField(lr_h, lr_w, n_channels, splat_scale=splat_scale, sigma_splat=sigma_splat,
                             sigma_psf=sigma_psf, margin=margin).to(dev)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        field = strategies.run_strategy(
            args.strategy, obs, shifts, lr_h=h, lr_w=w, n_channels=C,
            sigma_splat=args.sigma_splat, sigma_psf=psf, splat_scale=df, c2f_levels=c2f,
            lambda_tv=args.tv, huber_delta=args.huber_delta, n_iters=args.n_iters,
            lr_weights=args.lr_weights, lr_shifts=args.lr_shifts, n_outer=args.lbfgs_outer,
            adam_warmup=args.adam_warmup, adam_anneal=args.adam_anneal, adam_unreg=args.adam_unreg,
            learn_shifts=True, device=device, verbose=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0

        with torch.no_grad():
            hr_pred = field.render_hr_as_released(df) if args.no_fix_render else field.render_hr_exact(df)
        pred = hr_pred.permute(1, 2, 0).float().cpu().numpy()
        assert pred.shape == hr.shape, (pred.shape, hr.shape)

        true = true_shifts_lr(log)
        est = shifts.detach().cpu().numpy()
        meta = {"runtime_s": dt, "strategy": args.strategy, "init": args.init, "psf_lr_px": psf, "sigma_splat": args.sigma_splat,
                "c2f": list(c2f), "margin": margin, "fix_sign": not args.no_fix_sign,
                "fix_render": not args.no_fix_render, "device": device,
                "init_shifts_lr_px": init.numpy().tolist(), "final_shifts_lr_px": est.tolist(),
                "true_shifts_lr_px": None if true is None else true.tolist(),
                "note": "true shifts are the generator's HR-grid shifts /df; sign convention differs from "
                        "predict_lr, compare magnitudes/correlation, not raw values"}
        out = save_prediction(args.results, args.tag, ref, pred, meta)
        print(f"{args.tag:14s} {ref.scene} x{df}  {dt:.1f}s  -> {out}", flush=True)


if __name__ == "__main__":
    main()
