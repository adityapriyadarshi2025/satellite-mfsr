#!/usr/bin/env python3
"""SuperF (MSE / GNLL) and the paper's NIR baseline, using SuperF's own model and
training step (optimize.train_one_iteration), one fresh model per burst.

    python runners/run_superf.py --variant superf_mse  --superf_dir baselines/superf --device cuda:0
    python runners/run_superf.py --variant superf_gnll --superf_dir baselines/superf --device cuda:0
    python runners/run_superf.py --variant nir --iters 2000     # and again with --iters 5000 --tag nir5k

Why a wrapper instead of `python optimize.py`:
  * optimize.py saves matplotlib figures, not the raw prediction; we save pred.npy.
  * --model nir crashes at commit dff884c. The paper's NIR row is NIR "integrated in
    our framework": SuperF minus its three changes, i.e. an MLP-predicted affine per
    frame (no direct params), no fixed base frame, no supersampling (fit on the LR
    grid). That is exactly INR(use_direct_param_T=False, use_base_frame=False) with a
    training grid at scale 1, which is what --variant nir builds.
  * Paper GNLL = decoder outputs (T+1)*3 channels (RGB + one log-variance per frame).
    The single-sample CLI builds a 3-channel decoder, so --use_gnll alone gives
    all-zero log-variances. Here the decoder gets 3 + 3T outputs, as in the paper
    and in optimize.py's --multi_sample path.
  * --multi_sample reuses one decoder across scenes (warm start) and turns the
    per-frame colour projection off; the single-sample path keeps it on. The paper
    says it uses the projection, so it is on here (--no_color_shift to turn off).
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.bursts import add_common_args, select, save_prediction, pred_dir, true_shifts_lr  # noqa: E402

VARIANTS = {
    #               gnll   direct_T  base_frame  supersample
    "superf_mse":  (False, True,     True,       True),
    "superf_gnll": (True,  True,     True,       True),
    "nir":         (False, False,    False,      False),
}


def build_args():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--tag", default=None, help="results folder name (default: the variant)")
    p.add_argument("--superf_dir", default=os.path.join(os.environ.get("BASELINES", "baselines"), "superf"))
    p.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--fourier_scale", type=float, default=10.0, help="10 for satellite in the paper")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--no_color_shift", action="store_true")
    p.add_argument("--exact_area", action="store_true",
                   help="fairness check at non-integer scales: replace SuperF's 'area' downsampling (exact only at "
                        "integer factors) by the exact pixel average (common/exact_area.py); no effect at integer factors")
    return p.parse_args()


def seed_all(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_one(ref, args, device, mods):
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    SRData, get_decoder, get_input_projection, INR, train_one_iteration = mods
    use_gnll, direct_T, base_frame, supersample = VARIANTS[args.variant]
    n = args.num_frames

    seed_all(args.seed)
    data = SRData(data_dir=ref.path, num_samples=n, keep_in_memory=True,
                  scale_factor=float(ref.df) if supersample else 1.0, device=str(device))
    proj = get_input_projection("fourier", 2, args.proj_dim, device, args.fourier_scale)
    out_dim = 3 + 3 * n if use_gnll else 3
    dec = get_decoder("mlp", args.depth, args.proj_dim, args.hidden, output_dim=out_dim)
    model = INR(proj, dec, n, use_gnll=use_gnll, use_base_frame=base_frame,
                use_direct_param_T=direct_T, use_separate_ud=False,
                use_color_shift=not args.no_color_shift).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)
    loader = DataLoader(data, batch_size=1, shuffle=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    it, last = 0, {}
    while it < args.iters:
        for batch in loader:
            if it >= args.iters:
                break
            last = train_one_iteration(model, opt, batch, device)
            if not np.isfinite(last["total_loss"]):
                raise RuntimeError(f"non-finite loss at iter {it}: {last}")
            sched.step()
            it += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        hr_coords = data.get_hr_coordinates().unsqueeze(0).to(device)
        out, _ = model(hr_coords, torch.tensor([0], device=device), scale_factor=1, training=False)
        out = out * data.get_lr_std(0).to(device) + data.get_lr_mean(0).to(device)
    pred = out.squeeze(0).float().cpu().numpy()

    shifts = None
    if direct_T:
        shifts = [[float(v) for v in s.detach().cpu()] for s in model.shift_vectors]
    meta = {
        "runtime_s": train_s, "iters": args.iters, "variant": args.variant, "seed": args.seed,
        "fourier_scale": args.fourier_scale, "color_shift": not args.no_color_shift,
        "device": str(device), "final_losses": last, "exact_area": bool(getattr(args, "exact_area", False)),
        "learned_shift_params": shifts,
        "learned_shift_units": "SuperF coordinate units (image spans [0,1))",
        "true_shifts_lr_px": _first_n(true_shifts_lr(json.loads((Path(ref.path) / "transform_log.json").read_text())), n),
        "superf_commit": _git_head(args.superf_dir),
    }
    return pred, meta


def _first_n(arr, n):
    return None if arr is None else arr[:n].tolist()


def _git_head(d):
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(d), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = build_args()
    superf = Path(args.superf_dir).resolve()
    if not (superf / "optimize.py").exists():
        raise SystemExit(f"SuperF not found at {superf}. Run slurm/setup.sh (repos step)")
    sys.path.insert(0, str(superf))

    import torch
    from data import SRData
    from models.utils import get_decoder
    from input_projections.utils import get_input_projection
    from models.inr import INR
    from optimize import train_one_iteration
    mods = (SRData, get_decoder, get_input_projection, INR, train_one_iteration)

    if args.exact_area:
        from common.exact_area import install
        install()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tag = args.tag or args.variant
    print(f"{tag}: device {device}, {args.iters} iters" + (", exact area downsampling" if args.exact_area else ""))

    for ref in select(args):
        if args.skip_existing and (pred_dir(args.results, tag, ref) / "pred.npy").exists():
            continue
        pred, meta = run_one(ref, args, device, mods)
        out = save_prediction(args.results, tag, ref, pred, meta)
        ms = 1000 * meta["runtime_s"] / max(args.iters, 1)
        print(f"{tag:12s} {ref.scene} x{ref.df}  {meta['runtime_s']:.1f}s ({ms:.0f} ms/iter)  -> {out}", flush=True)


if __name__ == "__main__":
    main()
