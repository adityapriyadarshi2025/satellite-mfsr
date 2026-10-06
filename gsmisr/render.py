"""Differentiable 2D Gaussian rasteriser in pure PyTorch (CPU or GPU).

Scene (GaussianImage-style summation: no alpha, no depth sorting):

    f(x) = sum_k c_k * exp(-1/2 (x - mu_k)^T S_k^{-1} (x - mu_k))        c_k in R^C, unnormalised

`render` evaluates (f * N(0, p I))(x) at the centres of a regular pixel grid, where p is an extra
isotropic variance (sensor PSF, Gaussian stand-in for a pixel box, ...). Convolving an unnormalised
Gaussian of covariance S with a normalised kernel N(0, P) gives an unnormalised Gaussian with
covariance S + P and amplitude c * sqrt(det S / det(S + P)), so the mass c * 2*pi*sqrt(det S) is kept.

Units: scene coordinates are whatever the caller uses (gsmisr uses LR pixels of the reference frame,
x = column direction, y = row direction, pixel (i, j) covering x in [j, j+1), y in [i, i+1)).
A grid is (x0, y0, h, H, W): pixel (r, c) has its centre at (x0 + c*h, y0 + r*h).

Each Gaussian is evaluated on a (2R+1)^2 window around its nearest pixel; R comes from the largest
Gaussian (nsig standard deviations), so the truncation error is at most exp(-nsig^2/2) of a peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    h: float
    H: int
    W: int

    @staticmethod
    def pixel_centres(H: int, W: int, scale: float = 1.0) -> "Grid":
        """Grid of an image that covers scene [0, W/scale) x [0, H/scale) with pixel size 1/scale."""
        return Grid(0.5 / scale, 0.5 / scale, 1.0 / scale, H, W)


def covariance(scales: torch.Tensor, theta: torch.Tensor):
    """Sigma = R(theta) diag(s1^2, s2^2) R(theta)^T  ->  entries (a, b, d) with Sigma = [[a, b], [b, d]]."""
    c, s = torch.cos(theta), torch.sin(theta)
    v1, v2 = scales[:, 0] ** 2, scales[:, 1] ** 2
    a = c * c * v1 + s * s * v2
    b = c * s * (v1 - v2)
    d = s * s * v1 + c * c * v2
    return a, b, d


def max_std(a, b, d, extra_var=0.0) -> float:
    """Largest standard deviation over all Gaussians after adding extra_var (python float)."""
    with torch.no_grad():
        half_tr = 0.5 * (a + d)
        lam = half_tr + torch.sqrt(0.25 * (a - d) ** 2 + b * b)
        return float(torch.sqrt(lam.max() + float(extra_var)))


class _Splat(torch.autograd.Function):
    """The heavy part: sum of windowed Gaussians on a pixel grid, with a hand-written backward that
    recomputes the window instead of storing [N, P, P] intermediates (about 2x faster on CPU)."""

    @staticmethod
    def _window(u, v, ia, ib, id_, H, W, R):
        offs = torch.arange(-R, R + 1, device=u.device)
        cols = torch.round(u).long()[:, None] + offs[None, :]
        rows = torch.round(v).long()[:, None] + offs[None, :]
        dx = (cols.to(u.dtype) - u[:, None])[:, None, :]
        dy = (rows.to(u.dtype) - v[:, None])[:, :, None]
        q = ia[:, None, None] * dx * dx + 2.0 * ib[:, None, None] * dx * dy + id_[:, None, None] * dy * dy
        valid = ((rows >= 0) & (rows < H))[:, :, None] & ((cols >= 0) & (cols < W))[:, None, :]
        e = torch.exp(-0.5 * q) * valid
        idx = rows.clamp(0, H - 1)[:, :, None] * W + cols.clamp(0, W - 1)[:, None, :]
        return dx, dy, e, idx

    @staticmethod
    def forward(ctx, u, v, ia, ib, id_, amp, colors, H, W, R):
        with torch.no_grad():
            _, _, e, idx = _Splat._window(u, v, ia, ib, id_, H, W, R)
            w = amp[:, None, None] * e
            C = colors.shape[1]
            out = torch.zeros(H * W, C, dtype=u.dtype, device=u.device)
            out.index_add_(0, idx.reshape(-1), (w[..., None] * colors[:, None, None, :]).reshape(-1, C))
        ctx.save_for_backward(u, v, ia, ib, id_, amp, colors)
        ctx.dims = (H, W, R)
        return out.t().reshape(C, H, W)

    @staticmethod
    def backward(ctx, gout):
        u, v, ia, ib, id_, amp, colors = ctx.saved_tensors
        H, W, R = ctx.dims
        with torch.no_grad():
            dx, dy, e, idx = _Splat._window(u, v, ia, ib, id_, H, W, R)
            C = colors.shape[1]
            G = gout.reshape(C, H * W).t()[idx]                       # [N,P,P,C]
            gw = (G * colors[:, None, None, :]).sum(-1)               # dL/dw  [N,P,P]
            w = amp[:, None, None] * e
            d_col = (w[..., None] * G).sum((1, 2))
            d_amp = (e * gw).sum((1, 2))
            tq = -0.5 * w * gw                                        # dL/dq
            d_ia = (tq * dx * dx).sum((1, 2))
            d_ib = (tq * 2.0 * dx * dy).sum((1, 2))
            d_id = (tq * dy * dy).sum((1, 2))
            d_u = (tq * (-2.0 * ia[:, None, None] * dx - 2.0 * ib[:, None, None] * dy)).sum((1, 2))
            d_v = (tq * (-2.0 * ib[:, None, None] * dx - 2.0 * id_[:, None, None] * dy)).sum((1, 2))
        return d_u, d_v, d_ia, d_ib, d_id, d_amp, d_col, None, None, None


_RADII = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)


def render(mu: torch.Tensor, cov, colors: torch.Tensor, grid: Grid, extra_var=0.0,
           radius: int | None = None, nsig: float = 3.0, adaptive: bool = True) -> torch.Tensor:
    """mu [N,2] (x, y); cov = (a, b, d) each [N]; colors [N,C]  ->  image [C, H, W].

    radius: window half-size in pixels for every Gaussian (an upper bound when adaptive).
    adaptive: give each Gaussian its own window, ceil(nsig * its largest std / h), rounded up to a
    few fixed sizes and rendered group by group (most Gaussians are small, so this saves a lot).
    """
    a, b, d = cov
    p = extra_var if torch.is_tensor(extra_var) else torch.as_tensor(float(extra_var), dtype=mu.dtype, device=mu.device)
    ap, dp = a + p, d + p
    det_s = (a * d - b * b).clamp_min(1e-12)
    det_t = (ap * dp - b * b).clamp_min(1e-12)
    amp = torch.sqrt(det_s / det_t)                                    # [N]
    h = grid.h
    ia = dp / det_t * (h * h)
    ib = -b / det_t * (h * h)
    id_ = ap / det_t * (h * h)
    u = (mu[:, 0] - grid.x0) / h
    v = (mu[:, 1] - grid.y0) / h
    if radius is None:
        radius = max(1, int(math.ceil(nsig * max_std(a, b, d, float(p)) / h)))
    R = int(radius)
    if not adaptive:
        return _Splat.apply(u, v, ia, ib, id_, amp, colors, grid.H, grid.W, R)
    with torch.no_grad():
        half_tr = 0.5 * (ap + dp)
        lam = half_tr + torch.sqrt(0.25 * (ap - dp) ** 2 + b * b)
        rk = torch.ceil(nsig * torch.sqrt(lam) / h).clamp(1, R)
        sizes = torch.tensor([r for r in _RADII if r < R] + [R], device=mu.device, dtype=rk.dtype)
        cls = torch.bucketize(rk, sizes)                               # index of the smallest size >= rk
    out = None
    for ci in torch.unique(cls).tolist():
        sel = torch.nonzero(cls == ci, as_tuple=True)[0]
        Rc = int(sizes[ci])
        img = _Splat.apply(u[sel], v[sel], ia[sel], ib[sel], id_[sel], amp[sel], colors[sel], grid.H, grid.W, Rc)
        out = img if out is None else out + img
    return out


def render_dense_numpy(mu, scales, theta, colors, grid: Grid, extra_var=0.0):
    """Reference: exact evaluation at every pixel (no window). numpy float64. Returns [C,H,W]."""
    import numpy as np
    mu, scales, theta, colors = (np.asarray(t, dtype=np.float64) for t in (mu, scales, theta, colors))
    xs = grid.x0 + grid.h * np.arange(grid.W)
    ys = grid.y0 + grid.h * np.arange(grid.H)
    X, Y = np.meshgrid(xs, ys)
    out = np.zeros((colors.shape[1], grid.H, grid.W))
    for k in range(mu.shape[0]):
        c, s = np.cos(theta[k]), np.sin(theta[k])
        Rm = np.array([[c, -s], [s, c]])
        S = Rm @ np.diag(scales[k] ** 2) @ Rm.T
        T = S + extra_var * np.eye(2)
        Ti = np.linalg.inv(T)
        amp = np.sqrt(np.linalg.det(S) / np.linalg.det(T))
        dxs, dys = X - mu[k, 0], Y - mu[k, 1]
        q = Ti[0, 0] * dxs ** 2 + 2 * Ti[0, 1] * dxs * dys + Ti[1, 1] * dys ** 2
        out += amp * np.exp(-0.5 * q)[None] * colors[k][:, None, None]
    return out
