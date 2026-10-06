"""Exact pixel-average downsampling for PyTorch: a drop-in for F.interpolate(mode='area').

Why: SuperF shrinks its HR render to each LR frame with F.interpolate(..., mode='area'), which is
adaptive average pooling: output pixel i gets the plain mean of input pixels floor(i*r) ..
ceil((i+1)*r) - 1, r = in/out. That is the true pixel average only when r is an integer. At x2.56 or
x5.12 pixels that are only partly inside the footprint get full weight (on the SatSynthBurst HR images
the two differ by ~43 dB PSNR), while our rebuilt bursts (setup/make_arbscale.py, cv2 INTER_AREA) use
the true average. exact_area() weights each input pixel by its overlap with the output footprint,
separably in y and x: differentiable, any device, and identical to cv2 INTER_AREA.

    from common.exact_area import install; install()   # before SuperF's model runs

install() only changes 4-D 'area' calls that shrink by a non-integer factor; everything else
(integer factors, other modes) goes to PyTorch unchanged, so integer-scale runs are bit-identical.
"""
import math

import torch
import torch.nn.functional as F

_CACHE = {}


def area_matrix(n_in, n_out, device=None, dtype=torch.float32):
    """(n_out, n_in) matrix: row i = overlap of input pixel j with [i*r, (i+1)*r), divided by r."""
    key = (n_in, n_out, str(device), dtype)
    m = _CACHE.get(key)
    if m is None:
        r = n_in / n_out
        w = torch.zeros(n_out, n_in, dtype=torch.float64)
        for i in range(n_out):
            a, b = i * r, (i + 1) * r
            for j in range(int(math.floor(a)), min(n_in, int(math.ceil(b)))):
                ov = min(b, j + 1.0) - max(a, float(j))
                if ov > 0:
                    w[i, j] = ov / r
        m = w.to(device=device, dtype=dtype)
        _CACHE[key] = m
    return m


def exact_area(x, out_hw):
    """x: (..., H, W) -> (..., h, w); every output pixel is the exact mean over its footprint."""
    H, W = x.shape[-2:]
    h, w = out_hw
    ah = area_matrix(H, h, x.device, x.dtype)
    aw = area_matrix(W, w, x.device, x.dtype)
    return torch.matmul(torch.matmul(ah, x), aw.transpose(0, 1))


def _out_size(H, W, size, scale_factor):
    if size is not None:
        return (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    sf = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
    # same rule as PyTorch (recompute_scale_factor=None): floor(input_size * scale_factor) in double
    return int(math.floor(float(H) * float(sf[0]))), int(math.floor(float(W) * float(sf[1])))


def install():
    if getattr(F.interpolate, "_mfsr_exact_area", False):
        return
    orig = F.interpolate

    def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                    recompute_scale_factor=None, antialias=False):
        if mode == "area" and input.dim() == 4 and recompute_scale_factor is None:
            H, W = input.shape[-2:]
            h, w = _out_size(H, W, size, scale_factor)
            if 0 < h < H and 0 < w < W and (H % h or W % w):
                return exact_area(input, (h, w))
        return orig(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor, antialias=antialias)

    interpolate._mfsr_exact_area = True
    interpolate._mfsr_orig = orig
    F.interpolate = interpolate
