#!/usr/bin/env python3
"""SuperF (GNLL or MSE) plus the same total-variation prior as our TV version: the control for
"is the gain from the prior or from the Gaussians?". Same model, data, optimiser and schedule as
runners/run_superf.py; the only change is  loss += tv * TV(HR render)  at every iteration.

    python runners/run_superf_tv.py --variant superf_gnll --tv 0.1 --tag superf_gnll_tv --df 4 --aug light

The TV term uses the HR render that SuperF area-pools down to the current LR frame (captured from
its own forward pass, so no extra decoder pass), converted to intensity units with that frame's
per-channel std, with the formula gsmisr uses: mean of sqrt(gx^2 + gy^2 + eps^2).
The weight is NOT on the same scale as gsmisr's --tv: SuperF's data term is an unweighted MSE/GNLL
on per-frame standardised values. It is tuned separately (jobs/superf_tv.sh).
With --tv 0 the training step is SuperF's own train_one_iteration loss, re-implemented here
(tested to give the same prediction as run_superf.py).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runners"))
from common.bursts import add_common_args, select, save_prediction, pred_dir, true_shifts_lr  # noqa: E402
from run_superf import VARIANTS, seed_all, _first_n, _git_head  # noqa: E402

_STASH = {}


def build_args():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--variant", choices=["superf_mse", "superf_gnll"], required=True)
    p.add_argument("--tag", default=None, help="results folder name (default: <variant>_tv)")
    p.add_argument("--tv", type=float, default=0.0, help="weight of the TV prior (0 = plain SuperF)")
    p.add_argument("--tv_eps", type=float, default=1e-3)
    p.add_argument("--superf_dir", default=os.path.join(os.environ.get("BASELINES", "baselines"), "superf"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--fourier_scale", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--no_color_shift", action="store_true")
    return p.parse_args()


def install_capture():
    """Wrap torch.nn.functional.interpolate so the tensor SuperF area-pools (its HR render) is kept."""
    import torch.nn.functional as F
    if getattr(F.interpolate, "_mfsr_capture", False):
        return
    orig = F.interpolate

    def interpolate(input, *a, **k):
        if k.get("mode") == "area":
            _STASH["hr"] = input
        return orig(input, *a, **k)

    interpolate._mfsr_capture = True
    F.interpolate = interpolate


def train_step(model, opt, batch, device, use_gnll, tv, tv_eps):
    """SuperF's train_one_iteration loss (recon only, as in its total_loss) + optional TV."""
    import torch
    import torch.nn as nn
    model.train()
    inp = batch["input"].to(device)
    lr_target = batch["lr_target"].to(device)
    sid = batch["sample_id"].to(device)
    sf = batch["scale_factor"].to(device)
    opt.zero_grad()
    _STASH.clear()
    if use_gnll:
        output, _, var = model(inp, sid, scale_factor=1 / sf, lr_frames=lr_target)
        recon = nn.GaussianNLLLoss()(output, lr_target, var)
    else:
        output, _ = model(inp, sid, scale_factor=1 / sf, lr_frames=lr_target)
        recon = torch.square(output - lr_target).mean()
    loss = recon
    tv_v = 0.0
    if tv > 0:
        hr = _STASH.get("hr")
        if hr is None:
            raise RuntimeError("TV: no area-pooled HR render captured (SuperF code changed?)")
        std = batch["std"].to(device).reshape(hr.shape[0], -1)[:, :hr.shape[1]].reshape(hr.shape[0], -1, 1, 1)
        hr = hr * std                                   # intensity units of this frame
        gx = hr[:, :, :, 1:] - hr[:, :, :, :-1]
        gy = hr[:, :, 1:, :] - hr[:, :, :-1, :]
        tv_term = torch.sqrt(gx[:, :, :-1, :] ** 2 + gy[:, :, :, :-1] ** 2 + tv_eps ** 2).mean()
        loss = loss + tv * tv_term
        tv_v = float(tv_term)
    loss.backward()
    opt.step()
    return {"recon_loss": float(recon), "tv_term": tv_v, "total_loss": float(loss)}


def run_one(ref, args, device, mods):
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    SRData, get_decoder, get_input_projection, INR = mods
    use_gnll, direct_T, base_frame, supersample = VARIANTS[args.variant]
    n = args.num_frames
    seed_all(args.seed)
    data = SRData(data_dir=ref.path, num_samples=n, keep_in_memory=True,
                  scale_factor=float(ref.df) if supersample else 1.0, device=str(device))
    proj = get_input_projection("fourier", 2, args.proj_dim, device, args.fourier_scale)
    dec = get_decoder("mlp", args.depth, args.proj_dim, args.hidden, output_dim=3 + 3 * n if use_gnll else 3)
    model = INR(proj, dec, n, use_gnll=use_gnll, use_base_frame=base_frame,
                use_direct_param_T=direct_T, use_separate_ud=False,
                use_color_shift=not args.no_color_shift).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)
    loader = DataLoader(data, batch_size=1, shuffle=False)
    t0 = time.time()
    it, last = 0, {}
    while it < args.iters:
        for batch in loader:
            if it >= args.iters:
                break
            last = train_step(model, opt, batch, device, use_gnll, args.tv, args.tv_eps)
            if not np.isfinite(last["total_loss"]):
                raise RuntimeError(f"non-finite loss at iter {it}: {last}")
            sched.step()
            it += 1
    train_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        hr_coords = data.get_hr_coordinates().unsqueeze(0).to(device)
        out, _ = model(hr_coords, torch.tensor([0], device=device), scale_factor=1, training=False)
        out = out * data.get_lr_std(0).to(device) + data.get_lr_mean(0).to(device)
    pred = out.squeeze(0).float().cpu().numpy()
    shifts = [[float(v) for v in s.detach().cpu()] for s in model.shift_vectors] if direct_T else None
    meta = {
        "runtime_s": train_s, "iters": args.iters, "variant": args.variant, "seed": args.seed,
        "tv": args.tv, "tv_eps": args.tv_eps, "fourier_scale": args.fourier_scale,
        "color_shift": not args.no_color_shift, "device": str(device), "final_losses": last,
        "learned_shift_params": shifts, "learned_shift_units": "SuperF coordinate units (image spans [0,1))",
        "true_shifts_lr_px": _first_n(true_shifts_lr(json.loads((Path(ref.path) / "transform_log.json").read_text())), n),
        "superf_commit": _git_head(args.superf_dir),
    }
    return pred, meta


def main():
    args = build_args()
    superf = Path(args.superf_dir).resolve()
    if not (superf / "models" / "inr.py").exists():
        raise SystemExit(f"SuperF not found at {superf}")
    sys.path.insert(0, str(superf))
    import torch
    from data import SRData
    from models.utils import get_decoder
    from input_projections.utils import get_input_projection
    from models.inr import INR
    if args.tv > 0:
        install_capture()
    device = torch.device(args.device)
    tag = args.tag or f"{args.variant}_tv"
    print(f"{tag}: device {device}, {args.iters} iters, tv {args.tv}", flush=True)
    for ref in select(args):
        if args.skip_existing and (pred_dir(args.results, tag, ref) / "pred.npy").exists():
            continue
        pred, meta = run_one(ref, args, device, (SRData, get_decoder, get_input_projection, INR))
        out = save_prediction(args.results, tag, ref, pred, meta)
        print(f"{tag:16s} {ref.scene} x{ref.df}  {meta['runtime_s']:.1f}s  tv_term {meta['final_losses'].get('tv_term', 0):.4f}"
              f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
