"""Metric code shared by evaluate.py, with two scoring protocols.

- "code"  : what SuperF's released optimize.py reports (no crop, no colour match,
            torchmetrics PSNR/SSIM with data_range=1, LPIPS-VGG). Use it to check
            our numbers against SuperF's own metrics.json.
- "paper" : what the SuperF paper text says it did (Sec 4.2 + App. A.2): colour
            matching following Bhat et al. 2021 (DBSR), 16 px border crop,
            PSNR/SSIM, LPIPS-AlexNet. This is the protocol for the baseline table.

Colour matching is a numpy port of DBSR's match_colors (Gaussian sd 1.5, 7x7,
reflect pad, 5 px inner crop, 3x3 least squares). direction="dbsr" solves
pred @ C = gt, as in DBSR. direction="superf_compat" solves gt @ C = pred, as in
the colour-matching step of SuperF's handheld/evals.py.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure as _tm_ssim
    HAVE_TORCH = True
except Exception:  # torch missing: fall back to skimage SSIM
    HAVE_TORCH = False

_LPIPS = {}


def _gauss_1d(sd=1.5):
    ksz = int(4 * sd + 1)
    k = np.arange(-(ksz - 1) / 2, (ksz + 1) / 2)
    g = np.exp(-(k ** 2) / (2 * sd ** 2))
    return g / g.sum(), ksz


def _blur(im: np.ndarray, sd=1.5) -> np.ndarray:
    """Separable Gaussian blur, reflect padding (torch 'reflect' == numpy 'reflect')."""
    g, ksz = _gauss_1d(sd)
    p = ksz // 2
    x = np.pad(im, ((p, p), (p, p), (0, 0)), mode="reflect")
    x = sum(g[i] * x[i:i + im.shape[0], :, :] for i in range(ksz))
    x = sum(g[i] * x[:, i:i + im.shape[1], :] for i in range(ksz))
    return x


def match_colors(gt: np.ndarray, pred: np.ndarray, direction: str = "dbsr") -> np.ndarray:
    """Global 3x3 colour transform fitted on blurred images, applied to pred. HxWx3 in, HxWx3 out."""
    bi = 5
    r = _blur(gt.astype(np.float64))[bi:-bi, bi:-bi].reshape(-1, 3)
    q = _blur(pred.astype(np.float64))[bi:-bi, bi:-bi].reshape(-1, 3)
    if direction == "dbsr":
        C, *_ = np.linalg.lstsq(q, r, rcond=None)
    elif direction == "superf_compat":
        C, *_ = np.linalg.lstsq(r, q, rcond=None)
    else:
        raise ValueError(direction)
    return (pred.reshape(-1, 3).astype(np.float64) @ C).reshape(pred.shape).astype(np.float32)


def crop(im: np.ndarray, m: int) -> np.ndarray:
    return im[m:-m, m:-m] if m > 0 else im


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / mse)


def ssim(pred: np.ndarray, gt: np.ndarray, data_range=1.0):
    """Returns (value, implementation). torchmetrics if torch is present, else skimage (close, not identical)."""
    if HAVE_TORCH:
        a = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1)[None].float()
        b = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].float()
        return float(_tm_ssim(a, b, data_range=data_range)), "torchmetrics"
    from skimage.metrics import structural_similarity
    v = structural_similarity(gt, pred, channel_axis=-1, gaussian_weights=True, sigma=1.5,
                              use_sample_covariance=False, data_range=data_range)
    return float(v), "skimage"


def lpips_dist(pred: np.ndarray, gt: np.ndarray, net: str = "alex", device: str = "cpu"):
    """LPIPS with inputs mapped [0,1] -> [-1,1]. Needs torch + lpips (+ backbone weights cached)."""
    if not HAVE_TORCH:
        return float("nan")
    import lpips
    if net not in _LPIPS:
        _LPIPS[net] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    a = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1)[None].float().to(device) * 2 - 1
    b = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].float().to(device) * 2 - 1
    with torch.no_grad():
        return float(_LPIPS[net](a, b))


def score(pred: np.ndarray, gt: np.ndarray, protocol: str, *, use_lpips=False,
          colormatch="dbsr", border=16, device="cpu") -> dict:
    assert pred.shape == gt.shape, f"pred {pred.shape} vs gt {gt.shape}"
    if protocol == "code":
        p, g, net = pred, gt, "vgg"
    elif protocol == "paper":
        p = match_colors(gt, pred, colormatch)
        p, g, net = crop(p, border), crop(gt, border), "alex"
    else:
        raise ValueError(protocol)
    s, impl = ssim(p, g)
    out = {"psnr": psnr(p, g), "ssim": s, "ssim_impl": impl}
    out["lpips"] = lpips_dist(p, g, net, device) if use_lpips else float("nan")
    return out
