"""Shims that let GSASR's released inference code (github.com/ChrisDud0257/GSASR) run here without compiling.

1. `gscuda`: GSASR rasterises with a CUDA extension (utils/gs_cuda_dmax/gs.cu). `gs_render` below does the same
   arithmetic in PyTorch: every Gaussian is evaluated at the output pixel centres (normalised coordinates
   2*i/(n-1) - 1) inside a box of half-width dmax around its centre, multiplied by its colour and summed into the
   image. Point evaluation, no PSF, no pixel-area integration: that is GSASR's own image model, kept as it is.
   Forward only (inference).
2. `distutils.version`: GSASR's encoders import LooseVersion from distutils, which Python 3.12 no longer ships.
   It is only used in code paths EDSR and RDN don't take.

    from common.gsasr_shims import install
    install()          # before importing anything from GSASR; runs a small self-test of gs_render
"""

from __future__ import annotations

import math
import re
import sys
import types

import torch


def gs_render(sigmas, coords, colors, rendered_img, s, h, w, c, dmax, chunk=8192):
    """Same contract as gscuda.gs_render: adds the Gaussians into rendered_img [h, w, c] in place.

    sigmas [s, 3] = (sigma_x, sigma_y, rho) in normalised units; coords [s, 2] = (x, y) in [-1, 1];
    colors [s, c] (colour times opacity)."""
    H, W, C, dm = int(h), int(w), int(c), float(dmax)
    out = rendered_img.view(-1, C)
    dev = sigmas.device
    rx = int(math.floor(dm * (W - 1) / 2.0)) + 2          # window half-widths in pixels (a superset; the
    ry = int(math.floor(dm * (H - 1) / 2.0)) + 2          # exact |d| <= dmax test is applied below)
    ox = torch.arange(-rx, rx + 1, device=dev)
    oy = torch.arange(-ry, ry + 1, device=dev)
    for i0 in range(0, int(s), chunk):
        i1 = min(int(s), i0 + chunk)
        sx, sy, rho = sigmas[i0:i1, 0], sigmas[i0:i1, 1], sigmas[i0:i1, 2]
        x, y = coords[i0:i1, 0], coords[i0:i1, 1]
        col = colors[i0:i1]
        wi = torch.round((x + 1) * (W - 1) / 2.0).long()[:, None] + ox[None, :]     # [n, Kx]
        hi = torch.round((y + 1) * (H - 1) / 2.0).long()[:, None] + oy[None, :]     # [n, Ky]
        # as in the kernel: the pixel coordinate in double precision, stored as float, minus the centre
        dx = (2.0 * wi.double() / (W - 1) - 1.0).float() - x[:, None]
        dy = (2.0 * hi.double() / (H - 1) - 1.0).float() - y[:, None]
        okx = (dx <= dm) & (dx >= -dm) & (wi >= 0) & (wi < W)
        oky = (dy <= dm) & (dy >= -dm) & (hi >= 0) & (hi < H)
        a = 1.0 / sx / sx
        b = 2.0 * rho / sx / sy
        d = 1.0 / sy / sy
        k = -0.5 / (1.0 - rho * rho)
        DX, DY = dx[:, None, :], dy[:, :, None]
        q = a[:, None, None] * DX * DX - b[:, None, None] * DX * DY + d[:, None, None] * DY * DY
        v = torch.exp(q * k[:, None, None]) * (oky[:, :, None] & okx[:, None, :])
        idx = hi.clamp(0, H - 1)[:, :, None] * W + wi.clamp(0, W - 1)[:, None, :]
        out.index_add_(0, idx.reshape(-1), (v[..., None] * col[:, None, None, :]).reshape(-1, C))
    return rendered_img


def gs_render_backward(*args, **kwargs):
    raise NotImplementedError("GSASR shim: inference only (no backward)")


def _reference(sigmas, coords, colors, h, w, dmax):
    """The CUDA kernel written as plain loops (slow; for the self-test only)."""
    out = torch.zeros(h, w, colors.shape[1], dtype=torch.float32)
    for i in range(sigmas.shape[0]):
        sx, sy, rho = (float(v) for v in sigmas[i])
        x, y = (float(v) for v in coords[i])
        for hi in range(h):
            dy = float(torch.tensor(2.0 * hi / (h - 1) - 1.0, dtype=torch.float32)) - y
            if dy > dmax or dy < -dmax:
                continue
            for wi in range(w):
                dx = float(torch.tensor(2.0 * wi / (w - 1) - 1.0, dtype=torch.float32)) - x
                if dx > dmax or dx < -dmax:
                    continue
                v = dx * dx / (sx * sx) - 2 * rho * dx * dy / (sx * sy) + dy * dy / (sy * sy)
                out[hi, wi] += math.exp(-0.5 / (1 - rho * rho) * v) * colors[i]
    return out


def self_test(device="cpu"):
    """gs_render against the loop version of the kernel on a small random scene. Raises if they differ."""
    g = torch.Generator().manual_seed(0)
    n, h, w, dm = 40, 23, 31, 0.3
    sig = torch.stack([0.02 + 0.2 * torch.rand(n, generator=g), 0.02 + 0.2 * torch.rand(n, generator=g),
                       1.8 * torch.rand(n, generator=g) - 0.9], 1)
    xy = 2.4 * torch.rand(n, 2, generator=g) - 1.2                     # some centres outside the image
    col = torch.rand(n, 3, generator=g)
    ref = _reference(sig, xy, col, h, w, dm)
    img = torch.zeros(h, w, 3, device=device)
    gs_render(sig.to(device), xy.to(device), col.to(device), img, n, h, w, 3, dm, chunk=7)
    err = float((img.cpu() - ref).abs().max())
    if not err < 1e-4:
        raise RuntimeError(f"GSASR shim self-test failed: max abs difference {err:.2e}")
    return err


class _LooseVersion:
    """Minimal stand-in for distutils.version.LooseVersion (numeric parts only)."""

    def __init__(self, v):
        self.vstring = str(v)
        self.version = tuple(int(p) for p in re.findall(r"\d+", self.vstring))

    def _cmp(self, other):
        o = other if isinstance(other, _LooseVersion) else _LooseVersion(other)
        return (self.version > o.version) - (self.version < o.version)

    def __lt__(self, o): return self._cmp(o) < 0
    def __le__(self, o): return self._cmp(o) <= 0
    def __gt__(self, o): return self._cmp(o) > 0
    def __ge__(self, o): return self._cmp(o) >= 0
    def __eq__(self, o): return self._cmp(o) == 0
    def __repr__(self): return f"LooseVersion('{self.vstring}')"


def install(device="cpu", test=True):
    if "gscuda" not in sys.modules:
        m = types.ModuleType("gscuda")
        m.gs_render, m.gs_render_backward = gs_render, gs_render_backward
        sys.modules["gscuda"] = m
    try:
        import distutils.version  # noqa: F401
    except ModuleNotFoundError:
        pkg, ver = types.ModuleType("distutils"), types.ModuleType("distutils.version")
        ver.LooseVersion = _LooseVersion
        pkg.version = ver
        sys.modules["distutils"], sys.modules["distutils.version"] = pkg, ver
    if test:
        err = self_test(device)
        print(f"[gsasr] rasteriser shim self-test ok (max diff {err:.1e} vs the kernel's loops)", flush=True)


if __name__ == "__main__":
    print("max abs difference:", self_test("cuda" if torch.cuda.is_available() else "cpu"))
