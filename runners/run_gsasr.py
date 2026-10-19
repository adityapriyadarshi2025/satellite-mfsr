#!/usr/bin/env python3
"""GSASR (Chen et al., ICCV 2025): single-image arbitrary-scale SR with 2D Gaussians, using the authors' code and
their paper weights (trained on DIV2K with bicubic downsampling), applied to frame 0 of each burst. A single-image
baseline: it never sees the other frames, and its training degradation is not ours.

    python runners/run_gsasr.py --model EDSR --data_root data/satsynthburst --aug light --df 2 4 8
    python runners/run_gsasr.py --model RDN --data_root data/arbscale --results results_arb --aug none \
        --df 2.56 3.2 4 5.12 6.4

Needs once: bash slurm/setup.sh repos gsasr   (code at a pinned commit + the EDSR and RDN paper weights).
GSASR's rasteriser is a CUDA extension; common/gsasr_shims.py does the same arithmetic in PyTorch (no compiler
needed) and checks itself against a loop version of the kernel at start-up.
Inference follows the repo's inference_paper.py (no tiling): reflect-pad the input to a multiple of 12, predict
the Gaussians, render at floor(scale * padded size), crop to floor(scale * size), clamp to [0, 1].
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
from common.bursts import add_common_args, select, load_burst, save_prediction, pred_dir  # noqa: E402

BASE = os.environ.get("BASELINES", str(ROOT / "baselines"))


def build_args():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--model", type=str.upper, choices=["EDSR", "RDN"], default="EDSR",
                   help="encoder of the paper weights (EDSR-baseline or RDN)")
    p.add_argument("--tag", default=None, help="results folder name (default gsasr_<model>)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gsasr_dir", default=os.path.join(BASE, "gsasr"))
    p.add_argument("--weights_dir", default=os.path.join(BASE, "gsasr_weights"))
    p.add_argument("--dmax", type=float, default=0.1, help="rasterisation ratio r (the paper's 0.1)")
    return p.parse_args()


def load_gsasr(a, device):
    import torch
    from common.gsasr_shims import install
    install(device=str(device))
    sys.path.insert(0, str(Path(a.gsasr_dir).resolve()))
    from utils.fea2gs import Fea2GS                                       # noqa: E402  (GSASR's modules)
    from utils.gaussian_splatting import generate_2D_gaussian_splatting_step
    if a.model == "EDSR":
        from utils.edsrbaseline import EDSRNOUP as Encoder
    else:
        from utils.rdn import RDNNOUP as Encoder
    wdir = Path(a.weights_dir) / a.model
    enc, dec = Encoder(), Fea2GS()
    enc.load_state_dict(torch.load(wdir / "encoder.pth", map_location="cpu", weights_only=True)["params_ema"],
                        strict=True)
    dec.load_state_dict(torch.load(wdir / "decoder.pth", map_location="cpu", weights_only=True)["params_ema"],
                        strict=True)
    return enc.eval().to(device), dec.eval().to(device), generate_2D_gaussian_splatting_step


def upscale(img, scale, enc, dec, render, dmax, device):
    """img [h,w,3] RGB in [0,1] -> [floor(s*h), floor(s*w), 3]; same steps as inference_paper.py."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()[None].to(device)
    _, _, h, w = x.shape
    den = 12                                              # EDSR / RDN: the decoder's window size
    xp = F.pad(x, (0, (-w) % den, 0, (-h) % den), "reflect")
    out_size = (math.floor(scale * h), math.floor(scale * w))
    size_pad = torch.tensor([math.floor(scale * xp.shape[2]), math.floor(scale * xp.shape[3])])
    with torch.no_grad():
        feat = enc(xp)
        gs = dec(feat, torch.tensor(scale, dtype=torch.float32).unsqueeze(0).to(device))[0]
        y = render(gs_parameters=gs, sr_size=size_pad, scale=scale, sample_coords=None,
                   scale_modify=torch.tensor([scale, scale]), default_step_size=1.2, cuda_rendering=True,
                   mode="scale_modify", if_dmax=True, dmax_mode="fix", dmax=dmax)
    y = y[:, :out_size[0], :out_size[1]].clamp(0, 1)
    return y.permute(1, 2, 0).float().cpu().numpy(), int(gs.shape[0])


def main():
    a = build_args()
    import torch
    device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    tag = a.tag or f"gsasr_{a.model.lower()}"
    enc, dec, render = load_gsasr(a, device)
    print(f"{tag}: GSASR {a.model} on {device}, dmax {a.dmax}", flush=True)
    for ref in select(a):
        if a.skip_existing and (pred_dir(a.results, tag, ref) / "pred.npy").exists():
            continue
        frames, hr, _ = load_burst(ref, 1)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        pred, n_gs = upscale(frames[0], float(ref.df), enc, dec, render, a.dmax, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        assert pred.shape == hr.shape, (pred.shape, hr.shape)
        meta = {"runtime_s": dt, "device": str(device), "model": a.model, "dmax": a.dmax, "n_gaussians": n_gs,
                "input": "frame 0 only (single-image baseline)", "weights": f"GSASR paper {a.model}, DIV2K",
                "rasteriser": "common/gsasr_shims.py (PyTorch port of gs_cuda_dmax)"}
        if device.type == "cuda":
            meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            meta["gpu_name"] = torch.cuda.get_device_name(device)
        out = save_prediction(a.results, tag, ref, pred, meta)
        print(f"{tag:12s} {ref.scene} x{ref.df}  {dt:.2f}s  -> {out}", flush=True)


if __name__ == "__main__":
    main()
