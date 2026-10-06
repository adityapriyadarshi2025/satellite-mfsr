"""Gaussian multi-image super-resolution, per-scene fit (v0).

Scene: K anisotropic 2D Gaussians in the reference frame's LR pixel coordinates (x = column,
y = row; LR pixel (i, j) covers x in [j, j+1), y in [i, i+1)). Gaussians start on a regular grid
with `gpl` Gaussians per LR pixel per axis, over the image plus a `margin` (shifted frames see
past the edge). Each Gaussian has a bounded offset from its grid point, two scales (bounded, in
units of the grid spacing), a rotation and a colour.

Frame t (frame 0 = reference, fixed): rotation phi_t about the image centre + translation b_t
applied to the Gaussians (exact: mu -> R(mu - c) + c + b, theta -> theta + phi), then the sensor:
Gaussian PSF (variance added to every Gaussian), then the LR pixel box, done by evaluating at
m x m sub-pixel points per LR pixel and averaging (m = df reproduces SatSynthBurst's generator
exactly: HR samples -> blur -> df x df mean), then a per-frame colour transform (3x3 + offset).

Output at any scale s: the reference-frame scene evaluated at the s-grid pixel centres
("point", the HR-sample meaning SatSynthBurst uses) or averaged over each output pixel ("area",
Gaussian stand-in for the box, variance (1/s)^2/12).

Sign convention: SatSynthBurst frame t is output(x) = input(x + d_t), so its content moves by
-d_t and the learned b_t should approach -d_t. meta stores -b_t as `final_shifts_lr_px` so it is
directly comparable with the generator's (dx_pixels_lr, dy_pixels_lr).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field, replace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .render import Grid, covariance, render


@dataclass
class Config:
    gpl: int = 0                 # Gaussians per LR pixel per axis at the start; 0 = auto (2 up to x4, 4 at x8)
    margin: float = 2.0          # LR px of scene outside the image on each side
    s_lo: float = 0.2            # Gaussian std bounds, in units of the grid spacing 1/gpl
    s_hi: float = 1.0
    s_init: float = 0.6
    isotropic: bool = False      # round Gaussians (one scale, no rotation), for the ablation ladder
    max_off: float = 1.0         # max centre offset from the grid point, in grid spacings
    psf: float | None = None     # sensor PSF std in LR px; None = SatSynthBurst's 1/df^2
    sub: int | None = None       # sub-samples per LR pixel per axis; None = df (exact generator match)
    sub_box: bool = False        # also smooth each sub-sample over its sub-cell ((1/sub)^2/12)
    rotate: bool = True          # learn a rotation per frame
    color: str = "affine"        # per-frame colour model: affine (3x3 + offset) | gain (per channel) | none
    iters: int = 1000
    frames_per_iter: int = 8     # frames per iteration (always incl. frame 0); 0 = all
    lr_color: float = 1e-3
    lr_off: float = 2e-3
    lr_shape: float = 2e-3
    lr_shift: float = 1e-3
    lr_rot: float = 2e-4
    lr_cm: float = 1e-3
    shape_after: float = 0.1     # fraction of iterations before offsets/shapes start moving
    c2f_sigma: float = 0.8       # LR px blur of the residual at the start (coarse-to-fine), 0 = off
    c2f_frac: float = 0.3        # fraction of iterations over which it decays to 0
    init_shifts: str = "search"      # search (exhaustive NCC registration) | phasecorr | zero | given
    learn_shifts: bool = True
    noise: str = "residual"      # per-frame weighting by noise level, so noisy frames count less
                                 # (SatSynthBurst 'light' frames >= 2 are far noisier than frames 0-1):
                                 #   estimate: noise std from each frame's own pixels (MAD of the finest
                                 #             Haar detail), weights 1/std^2, fixed for the whole fit
                                 #   frame:    learn one std per frame (likelihood); can go "lazy": a
                                 #             misaligned frame gets a big std and is then ignored
                                 #   residual: start from `estimate`; once the coarse-to-fine phase is over,
                                 #             re-estimate every `reweight_every` iterations from each
                                 #             frame's own residual (iteratively reweighted least squares)
                                 #   none:     plain MSE
    noise_floor: float = 0.003   # lower bound on the estimated std (LR units) so no frame dominates
    reweight_every: int = 50
    lr_noise: float = 1e-2
    align_warmup: int = 30       # iterations that only register the frames against the starting scene
    lr_shift_warmup: float = 1e-2
    tv: float = 0.0              # weight of a total-variation prior on the HR render at the output scale
                                 # (Charbonnier). The data can't pin down every HR detail (2 clean frames +
                                 # 14 noisy ones at x2, 22% sub-pixel coverage at x8), and without a prior
                                 # the Gaussians fill that freedom with ripple; classical MISR uses
                                 # (bilateral) TV for the same reason.
    tv_eps: float = 1e-3
    nsig: float = 3.0
    hr_mode: str = "point"       # point | area
    seed: int = 0
    log_every: int = 100


def _gauss_kernel1d(sigma, device, dtype):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img [B,C,H,W], separable Gaussian, reflect padding."""
    if sigma <= 0.05:
        return img
    k = _gauss_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    C = img.shape[1]
    r = min(r, img.shape[-1] - 1, img.shape[-2] - 1)
    k = k[(k.numel() - 1) // 2 - r:(k.numel() - 1) // 2 + r + 1]
    k = k / k.sum()
    x = F.pad(img, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def phase_corr_shifts(frames: np.ndarray) -> np.ndarray:
    """[T,h,w,C] -> [T,2] OpenCV phase-correlation shifts of each frame vs frame 0 (sign as returned)."""
    g = frames.mean(axis=-1).astype(np.float64)
    win = cv2.createHanningWindow((g.shape[2], g.shape[1]), cv2.CV_64F)
    out = np.zeros((g.shape[0], 2))
    for t in range(1, g.shape[0]):
        (dx, dy), _ = cv2.phaseCorrelate(g[0], g[t], win)
        out[t] = (dx, dy)
    return out


def estimate_noise(frames: np.ndarray) -> np.ndarray:
    """[T,h,w,C] -> [T] noise std per frame: median absolute finest diagonal Haar detail / 0.6745
    (Donoho). White noise of std s gives s; sparse edges barely move the median."""
    f = frames[:, : frames.shape[1] // 2 * 2, : frames.shape[2] // 2 * 2].astype(np.float64)
    hh = (f[:, 0::2, 0::2] - f[:, 0::2, 1::2] - f[:, 1::2, 0::2] + f[:, 1::2, 1::2]) / 2.0
    return np.median(np.abs(hh).reshape(f.shape[0], -1), axis=1) / 0.6745


def register_search(frames: np.ndarray, coarse: float = 1.5, step: float = 0.25, fine_step: float = 0.05,
                    up: int = 4, blur_sigma: float = 1.0, border: int = 3) -> np.ndarray:
    """Exhaustive registration of every frame against the reference, in the model's own convention.

    Returns b [T,2] (LR px) such that frame t ~ scene moved by +b_t (b_0 = 0). The reference is
    frame 0 upsampled (bicubic, x`up`); a candidate shift b predicts frame t as the LR-pixel box
    average of that reference around (pixel centre - b). Candidates on a coarse grid
    [-coarse, coarse]^2, then a fine grid around the best one; score = normalised cross-correlation
    of the Gaussian-blurred prediction and frame (robust to brightness/contrast changes)."""
    g = frames.mean(-1).astype(np.float32)                       # [T,h,w]
    T, h, w = g.shape
    ref = cv2.resize(cv2.copyMakeBorder(g[0], 4, 4, 4, 4, cv2.BORDER_REFLECT), ((w + 8) * up, (h + 8) * up),
                     interpolation=cv2.INTER_CUBIC)
    Bx = torch.from_numpy(ref)[None, None]
    B = F.avg_pool2d(Bx, up, stride=1)[0, 0]                      # B[k] = box centred at k/up + 0.5 - 4
    Hb, Wb = B.shape
    obs = torch.from_numpy(np.stack([cv2.GaussianBlur(x, (0, 0), blur_sigma) for x in g]))

    def predict(cands):                                          # cands [K,2] (bx, by) -> [K,h,w]
        jj = torch.arange(w, dtype=torch.float32)
        ii = torch.arange(h, dtype=torch.float32)
        kx = up * (jj[None, :] + 4 - cands[:, :1])               # [K,w] fractional index into B
        ky = up * (ii[None, :] + 4 - cands[:, 1:])               # [K,h]
        gx = (2 * kx + 1) / Wb - 1                               # align_corners=False normalisation
        gy = (2 * ky + 1) / Hb - 1
        grid = torch.stack([gx[:, None, :].expand(-1, h, -1), gy[:, :, None].expand(-1, -1, w)], -1)
        out = F.grid_sample(B[None, None].expand(len(cands), 1, Hb, Wb), grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[:, 0]
        k = _gauss_kernel1d(blur_sigma, out.device, out.dtype)
        r = (k.numel() - 1) // 2
        out = F.conv2d(F.pad(out[:, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        out = F.conv2d(F.pad(out, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
        return out[:, 0]

    def ncc(P, Y):                                               # P [K,h,w], Y [h,w] -> [K]
        P = P[:, border:-border, border:-border].reshape(len(P), -1)
        Y = Y[border:-border, border:-border].reshape(-1)
        P = P - P.mean(1, keepdim=True)
        Y = Y - Y.mean()
        return (P @ Y) / (P.norm(dim=1) * Y.norm() + 1e-12)

    def grid_of(centre, half, st):
        v = torch.arange(-half, half + 1e-6, st)
        X, Y = torch.meshgrid(v, v, indexing="xy")
        return torch.stack([X.reshape(-1), Y.reshape(-1)], 1) + centre

    out = np.zeros((T, 2))
    coarse_c = grid_of(torch.zeros(2), coarse, step)
    Pc = predict(coarse_c)
    for t in range(1, T):
        best = coarse_c[int(torch.argmax(ncc(Pc, obs[t])))]
        fc = grid_of(best, step, fine_step)
        out[t] = fc[int(torch.argmax(ncc(predict(fc), obs[t])))].numpy()
    return out


class GaussMISR(nn.Module):
    def __init__(self, h: int, w: int, C: int, T: int, cfg: Config):
        super().__init__()
        self.h, self.w, self.C, self.T, self.cfg = h, w, C, T, cfg
        g = 1.0 / cfg.gpl
        self.g = g
        nx = int(round((w + 2 * cfg.margin) * cfg.gpl))
        ny = int(round((h + 2 * cfg.margin) * cfg.gpl))
        self.nx, self.ny = nx, ny
        xs = (torch.arange(nx) + 0.5) * g - cfg.margin
        ys = (torch.arange(ny) + 0.5) * g - cfg.margin
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("mu0", torch.stack([X.reshape(-1), Y.reshape(-1)], 1))
        N = nx * ny
        self.off = nn.Parameter(torch.zeros(N, 2))
        span = cfg.s_hi - cfg.s_lo
        frac = 0.5 if span <= 1e-9 else min(max((cfg.s_init - cfg.s_lo) / span, 1e-3), 1 - 1e-3)
        self.scale_raw = nn.Parameter(torch.full((N, 2), math.log(frac / (1 - frac))))
        self.theta = nn.Parameter(torch.zeros(N))
        self.color = nn.Parameter(torch.zeros(N, C))
        self.shift = nn.Parameter(torch.zeros(T - 1, 2))
        self.rot = nn.Parameter(torch.zeros(T - 1), requires_grad=cfg.rotate)
        self.cmat = nn.Parameter(torch.eye(C).repeat(T - 1, 1, 1), requires_grad=cfg.color == "affine")
        self.cgain = nn.Parameter(torch.ones(T - 1, C), requires_grad=cfg.color == "gain")
        self.coff = nn.Parameter(torch.zeros(T - 1, C), requires_grad=cfg.color in ("affine", "gain"))
        self.centre = torch.tensor([w / 2.0, h / 2.0])

    # ---- scene -------------------------------------------------------------------------------
    def means(self):
        return self.mu0 + self.cfg.max_off * self.g * torch.tanh(self.off)

    def scales(self):
        c = self.cfg
        raw = self.scale_raw[:, :1].expand(-1, 2) if c.isotropic else self.scale_raw
        return self.g * (c.s_lo + (c.s_hi - c.s_lo) * torch.sigmoid(raw))

    def radius(self, extra_var: float, h_grid: float) -> int:
        smax = self.g * self.cfg.s_hi
        return max(1, int(math.ceil(self.cfg.nsig * math.sqrt(smax * smax + extra_var) / h_grid)))

    # ---- frames ------------------------------------------------------------------------------
    def frame_geometry(self, t: int):
        mu, theta = self.means(), self.theta
        if t == 0:
            return mu, theta
        phi = self.rot[t - 1]
        c, s = torch.cos(phi), torch.sin(phi)
        ctr = self.centre.to(mu)
        d = mu - ctr
        rot = torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)
        return rot + ctr + self.shift[t - 1], theta + phi

    def apply_color(self, img: torch.Tensor, t: int) -> torch.Tensor:
        if t == 0 or self.cfg.color == "none":
            return img
        if self.cfg.color == "gain":
            return img * self.cgain[t - 1][:, None, None] + self.coff[t - 1][:, None, None]
        return torch.einsum("cd,dhw->chw", self.cmat[t - 1], img) + self.coff[t - 1][:, None, None]

    def render_frame(self, t: int, df, sub: int | None = None, sub_box: bool | None = None) -> torch.Tensor:
        """Frame t on its LR grid. LR pixel = mean of m x m sub-pixel samples (m = df reproduces the
        integer-scale generator exactly). For a non-integer df the LR pixel covers a fractional number
        of HR pixels, so the default is m = ceil(df) with each sample smoothed over its sub-cell
        (a continuous box), which is what an exact area average of the HR image amounts to."""
        cfg = self.cfg
        integral = float(df).is_integer()
        m = sub or cfg.sub or (int(df) if integral else int(math.ceil(df)))
        box = (cfg.sub_box or not integral) if sub_box is None else sub_box
        psf = cfg.psf if cfg.psf is not None else 1.0 / df ** 2
        extra = psf * psf + ((1.0 / m) ** 2 / 12.0 if box else 0.0)
        mu, theta = self.frame_geometry(t)
        grid = Grid.pixel_centres(self.h * m, self.w * m, float(m))
        img = render(mu, covariance(self.scales(), theta), self.color, grid, extra_var=extra,
                     radius=self.radius(extra, 1.0 / m))
        if m > 1:
            img = F.avg_pool2d(img[None], m)[0]
        return self.apply_color(img, t)

    def render_hr(self, scale: float) -> torch.Tensor:
        H, W = int(round(self.h * scale)), int(round(self.w * scale))
        extra = (1.0 / scale) ** 2 / 12.0 if self.cfg.hr_mode == "area" else 0.0
        grid = Grid.pixel_centres(H, W, float(scale))
        return render(self.means(), covariance(self.scales(), self.theta), self.color, grid,
                      extra_var=extra, radius=self.radius(extra, 1.0 / scale))

    # ---- init --------------------------------------------------------------------------------
    @torch.no_grad()
    def init_colors(self, base: np.ndarray):
        """base [h,w,C] in [0,1]: bicubic-upsampled reference frame at the Gaussian centres,
        divided by the sum of unit Gaussians at a grid point (so the start renders ~ bicubic)."""
        m = int(math.ceil(self.cfg.margin))
        pad = cv2.copyMakeBorder(base.astype(np.float32), m, m, m, m, cv2.BORDER_REPLICATE)
        # pad covers [-m, w+m); the Gaussian grid covers [-margin, w+margin) with margin <= m
        up = cv2.resize(pad, (int(round((self.w + 2 * m) * self.cfg.gpl)), int(round((self.h + 2 * m) * self.cfg.gpl))),
                        interpolation=cv2.INTER_CUBIC)
        k0 = int(round((m - self.cfg.margin) * self.cfg.gpl))
        up = up[k0:k0 + self.ny, k0:k0 + self.nx]
        s0 = self.cfg.s_init
        i = np.arange(-8, 9)
        norm = np.exp(-0.5 * (i[:, None] ** 2 + i[None, :] ** 2) / s0 ** 2).sum()
        self.color.copy_(torch.from_numpy(up.reshape(-1, self.C) / norm).to(self.color))


@dataclass
class FitResult:
    hr: np.ndarray                      # [H,W,C]
    model: GaussMISR
    runtime_s: float
    ms_per_iter: float
    final_loss: float
    init_shifts_lr_px: list
    final_shifts_lr_px: list            # -b_t, comparable with the generator's (dx, dy)
    final_rot_rad: list
    history: list = field(default_factory=list)
    noise_std: list = field(default_factory=list)   # learned per-frame noise std (LR units), if noise="frame"


def fit(frames: np.ndarray, df: int, cfg: Config, out_scale: float | None = None, device: str = "cpu",
        verbose: bool = True, given_d: np.ndarray | None = None, true_d: np.ndarray | None = None) -> FitResult:
    """frames [T,h,w,C] float in [0,1], frame 0 = reference.
    given_d [T,2]: starting shifts in the generator's convention (content of frame t moved by -d_t),
    used when cfg.init_shifts == "given" (oracle / diagnostic runs).
    true_d [T,2]: only for logging the shift error while fitting (never used by the fit)."""
    torch.manual_seed(cfg.seed)
    if cfg.gpl == 0:
        cfg = replace(cfg, gpl=2 if df <= 4 else 4)
    T, h, w, C = frames.shape

    def shift_err():
        if true_d is None:
            return ""
        est = -model.shift.detach().cpu().numpy()
        return f"  shift err {np.abs(est - np.asarray(true_d)[1:]).mean():.3f} LR px"
    obs = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().to(device)
    model = GaussMISR(h, w, C, T, cfg).to(device)
    model.centre = model.centre.to(device)
    model.init_colors(frames[0])

    init = np.zeros((T, 2))
    if cfg.init_shifts == "given":
        if given_d is None:
            raise ValueError("init_shifts='given' needs given_d")
        init = np.asarray(given_d, dtype=np.float64)
        with torch.no_grad():
            model.shift.copy_(torch.from_numpy(-init[1:]).float())
    elif cfg.init_shifts == "search" and T > 1:
        b = register_search(frames)
        with torch.no_grad():
            model.shift.copy_(torch.from_numpy(b[1:]).float())
        init = -b
        if verbose:
            print(f"[gsmisr] search registration init{shift_err()}", flush=True)
    elif cfg.init_shifts == "phasecorr" and T > 1:
        pc = phase_corr_shifts(frames)
        with torch.no_grad():
            losses = []
            for sign in (+1.0, -1.0):
                model.shift.copy_(torch.from_numpy(sign * pc[1:]).float())
                losses.append(sum(float(((model.render_frame(t, df) - obs[t]) ** 2).mean())
                                  for t in range(1, T)))
            sign = +1.0 if losses[0] <= losses[1] else -1.0
            model.shift.copy_(torch.from_numpy(sign * pc[1:]).float())
            init = -sign * pc                     # in the generator's (dx, dy) convention
        if verbose:
            print(f"[gsmisr] phase-correlation init: sign {sign:+.0f} (loss {min(losses):.3e} vs {max(losses):.3e})"
                  f"{shift_err()}", flush=True)

    # ---- per-frame weights from the noise level ---------------------------------------------
    sig_est = estimate_noise(frames)
    if cfg.noise in ("estimate", "residual"):
        s = np.maximum(sig_est, cfg.noise_floor)
        wts = 1.0 / s ** 2
        wts = wts / wts.mean()
    else:
        wts = np.ones(T)
    wts_t = torch.tensor(wts, dtype=torch.float32, device=device)
    if verbose:
        print(f"[gsmisr] noise std per frame (data estimate): {np.round(sig_est, 4).tolist()}", flush=True)

    # ---- stage 1: register every frame against the starting scene (bicubic of frame 0) -------
    # cheap render (LR grid, Gaussian pixel box), blurred residual, only frame parameters move
    t0 = time.time()
    if cfg.align_warmup > 0 and cfg.learn_shifts and T > 1:
        wg = [{"params": [model.shift], "lr": cfg.lr_shift_warmup}]
        if cfg.rotate:
            wg.append({"params": [model.rot], "lr": cfg.lr_rot})
        if cfg.color == "affine":
            wg.append({"params": [model.cmat, model.coff], "lr": cfg.lr_cm * 3})
        elif cfg.color == "gain":
            wg.append({"params": [model.cgain, model.coff], "lr": cfg.lr_cm * 3})
        wopt = torch.optim.Adam(wg)
        wbase = [g["lr"] for g in wopt.param_groups]
        model.color.requires_grad_(False)
        for it in range(cfg.align_warmup):
            cos = 0.5 * (1 + math.cos(math.pi * it / max(1, cfg.align_warmup - 1)))
            for gi, g in enumerate(wopt.param_groups):
                g["lr"] = wbase[gi] * (0.05 + 0.95 * cos)
            sig = max(cfg.c2f_sigma, 0.5) * (0.5 + 0.5 * cos)
            wopt.zero_grad(set_to_none=True)
            loss = 0.0
            for t in range(1, T):
                res = blur((model.render_frame(t, df, sub=1, sub_box=True) - obs[t])[None], sig)
                loss = loss + (res ** 2).mean()
            (loss / (T - 1)).backward()
            wopt.step()
        model.color.requires_grad_(True)
        if verbose:
            print(f"[gsmisr] registration warm-up: {cfg.align_warmup} it, {time.time() - t0:.0f} s{shift_err()}", flush=True)

    # ---- stage 2: everything together ---------------------------------------------------------
    groups = [
        {"params": [model.color], "lr": cfg.lr_color},
        {"params": [model.off], "lr": cfg.lr_off},
        {"params": [model.scale_raw, model.theta], "lr": cfg.lr_shape},
    ]
    if cfg.learn_shifts:
        groups.append({"params": [model.shift], "lr": cfg.lr_shift})
    if cfg.rotate:
        groups.append({"params": [model.rot], "lr": cfg.lr_rot})
    if cfg.color == "affine":
        groups.append({"params": [model.cmat, model.coff], "lr": cfg.lr_cm})
    elif cfg.color == "gain":
        groups.append({"params": [model.cgain, model.coff], "lr": cfg.lr_cm})
    logsig = torch.full((T,), math.log(0.01), device=device, requires_grad=cfg.noise == "frame")
    if cfg.noise == "frame":
        groups.append({"params": [logsig], "lr": cfg.lr_noise})
    opt = torch.optim.Adam(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    hold = int(cfg.shape_after * cfg.iters)
    gen = torch.Generator().manual_seed(cfg.seed)
    hist = []
    t1 = time.time()
    loss_v = float("nan")
    for it in range(cfg.iters):
        frac = it / max(1, cfg.iters - 1)
        cos = 0.5 * (1 + math.cos(math.pi * frac))
        for gidx, g in enumerate(opt.param_groups):
            lr = base_lrs[gidx] * (0.02 + 0.98 * cos)
            if gidx in (1, 2) and it < hold:      # offsets and shapes wait until colours/alignment settle
                lr = 0.0
            g["lr"] = lr
        sig = cfg.c2f_sigma * max(0.0, 1.0 - it / max(1, cfg.c2f_frac * cfg.iters))
        if (cfg.noise == "residual" and sig == 0.0 and it % cfg.reweight_every == 0):
            with torch.no_grad():
                mses = np.array([float(((model.render_frame(t_, df) - obs[t_]) ** 2).mean()) for t_ in range(T)])
            s = np.maximum(np.sqrt(mses), cfg.noise_floor)
            w_ = 1.0 / s ** 2
            wts_t = torch.tensor(w_ / w_.mean(), dtype=torch.float32, device=device)
            sig_est = s
        if cfg.frames_per_iter and cfg.frames_per_iter < T:
            ts = [0] + (1 + torch.randperm(T - 1, generator=gen)[:cfg.frames_per_iter - 1]).tolist()
        else:
            ts = list(range(T))
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        mse_sum = 0.0
        for t in ts:
            res = (model.render_frame(t, df) - obs[t])[None]
            res = blur(res, sig) if sig > 0.05 else res
            mse = (res ** 2).mean()
            mse_sum = mse_sum + float(mse)
            if cfg.noise == "frame":
                ls = logsig[t].clamp_min(math.log(1e-3))
                loss = loss + 0.5 * mse * torch.exp(-2 * ls) + ls
            else:
                loss = loss + wts_t[t] * mse
        loss = loss / len(ts)
        if cfg.tv > 0:
            hr_img = model.render_hr(df)
            gx = hr_img[:, :, 1:] - hr_img[:, :, :-1]
            gy = hr_img[:, 1:, :] - hr_img[:, :-1, :]
            tv_term = torch.sqrt(gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2 + cfg.tv_eps ** 2).mean()
            loss = loss + cfg.tv * tv_term
        loss.backward()
        opt.step()
        loss_v = mse_sum / len(ts)
        if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
            el = time.time() - t1
            print(f"[gsmisr] it {it:5d}  mse {loss_v:.3e}  c2f {sig:.2f}  {1000 * el / (it + 1):.0f} ms/iter{shift_err()}", flush=True)
            hist.append((it, loss_v))
    dt = time.time() - t0
    with torch.no_grad():
        hr = model.render_hr(out_scale or df).permute(1, 2, 0).cpu().numpy()
        est = np.zeros((T, 2))
        est[1:] = -model.shift.detach().cpu().numpy()
        rot = np.zeros(T)
        rot[1:] = model.rot.detach().cpu().numpy()
    res_ = FitResult(hr=hr, model=model, runtime_s=dt, ms_per_iter=1000 * (time.time() - t1) / max(1, cfg.iters),
                     final_loss=loss_v, init_shifts_lr_px=init.tolist(), final_shifts_lr_px=est.tolist(),
                     final_rot_rad=rot.tolist(), history=hist)
    res_.noise_std = (torch.exp(logsig.detach()).cpu().numpy().tolist() if cfg.noise == "frame"
                      else sig_est.tolist())
    return res_


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)
