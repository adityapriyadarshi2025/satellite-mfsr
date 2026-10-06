"""Checks for the Gaussian rasteriser and the burst forward model.

    python -m pytest -q gsmisr/tests          (or: python gsmisr/tests/test_render.py)

1. windowed torch renderer == dense numpy evaluation (anisotropic, rotated, extra blur, any grid)
2. convolving with the extra blur keeps each Gaussian's mass
3. a pure translation of the Gaussians == the same image shifted by whole pixels
4. render with a Gaussian PSF == numerically blurring a fine render (checks the amplitude formula)
5. gradients match central finite differences (float64)
6. exhaustive registration recovers known sub-pixel shifts
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsmisr.render import Grid, covariance, render, render_dense_numpy  # noqa: E402


def _random_scene(n, rng, extent=8.0, smin=0.2, smax=1.2):
    mu = rng.uniform(1.0, extent - 1.0, (n, 2))
    scales = rng.uniform(smin, smax, (n, 2))
    theta = rng.uniform(-math.pi, math.pi, n)
    colors = rng.normal(0, 1, (n, 3))
    return mu, scales, theta, colors


def _torch(*arrs, dtype=torch.float64):
    return [torch.tensor(a, dtype=dtype) for a in arrs]


def test_matches_dense():
    rng = np.random.default_rng(0)
    mu, sc, th, col = _random_scene(40, rng)
    for grid, extra in [(Grid.pixel_centres(8, 8, 1.0), 0.0), (Grid.pixel_centres(32, 32, 4.0), 0.1),
                        (Grid(0.3, -0.2, 0.37, 25, 21), 0.02)]:
        ref = render_dense_numpy(mu, sc, th, col, grid, extra)
        m, s, t, c = _torch(mu, sc, th, col)
        out = render(m, covariance(s, t), c, grid, extra_var=extra, nsig=5.0).numpy()
        err = np.abs(out - ref).max() / np.abs(ref).max()
        assert err < 1e-5, (grid, extra, err)


def test_mass_preserved_under_blur():
    rng = np.random.default_rng(1)
    mu, sc, th, col = _random_scene(5, rng, extent=16.0, smin=0.3, smax=0.8)
    mu = mu * 0 + 8.0 + rng.uniform(-1, 1, mu.shape)
    g = Grid.pixel_centres(64, 64, 4.0)   # 16 x 16 scene units, fine pixels
    m, s, t, c = _torch(mu, sc, th, col)
    a = render(m, covariance(s, t), c, g, extra_var=0.0, nsig=6.0).sum((1, 2))
    b = render(m, covariance(s, t), c, g, extra_var=0.5, nsig=6.0).sum((1, 2))
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (a, b)


def test_translation_equals_pixel_shift():
    rng = np.random.default_rng(2)
    mu, sc, th, col = _random_scene(30, rng, extent=12.0)
    g = Grid.pixel_centres(48, 48, 4.0)
    m, s, t, c = _torch(mu, sc, th, col)
    base = render(m, covariance(s, t), c, g, nsig=5.0)
    moved = render(m + torch.tensor([0.5, 0.25], dtype=m.dtype), covariance(s, t), c, g, nsig=5.0)
    # +0.5 scene units in x = +2 pixels, +0.25 in y = +1 pixel at 4 px per unit
    assert torch.allclose(moved[:, 1:, 2:], base[:, :-1, :-2], atol=1e-8)


def test_psf_amplitude_matches_numerical_blur():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    mu, sc, th, col = _random_scene(6, rng, extent=10.0, smin=0.3, smax=0.7)
    mu = 3.0 + rng.uniform(0, 4, mu.shape)
    fine = Grid.pixel_centres(160, 160, 16.0)       # 10 x 10 units at 16 px/unit
    m, s, t, c = _torch(mu, sc, th, col)
    sharp = render(m, covariance(s, t), c, fine, nsig=6.0).numpy()
    psf = 0.3                                         # std in scene units
    num = np.stack([gaussian_filter(ch, psf * 16.0, mode="constant", truncate=6.0) for ch in sharp])
    ana = render(m, covariance(s, t), c, fine, extra_var=psf ** 2, nsig=6.0).numpy()
    err = np.abs(num - ana)[:, 20:-20, 20:-20].max() / np.abs(ana).max()
    assert err < 2e-3, err


def test_gradients_match_finite_differences():
    """Autograd vs central differences for a random scalar of the image (float64)."""
    rng = np.random.default_rng(4)
    mu, sc, th, col = _random_scene(3, rng, extent=5.0, smin=0.4, smax=0.9)
    g = Grid(0.25, 0.25, 0.5, 10, 10)
    wts = torch.tensor(rng.normal(0, 1, (3, 10, 10)), dtype=torch.float64)

    def f(m_, s_, t_, c_):
        return (render(m_, covariance(s_, t_), c_, g, extra_var=0.05, radius=6) * wts).sum()

    xs = [x.requires_grad_(True) for x in _torch(mu, sc, th, col)]
    f(*xs).backward()
    eps = 1e-6
    for i, x in enumerate(xs):
        num = torch.zeros_like(x)
        flat = x.detach().clone().reshape(-1)
        for j in range(flat.numel()):
            args = [y.detach().clone() for y in xs]
            plus, minus = flat.clone(), flat.clone()
            plus[j] += eps
            minus[j] -= eps
            args[i] = plus.reshape(x.shape)
            fp = f(*args).item()
            args[i] = minus.reshape(x.shape)
            fm = f(*args).item()
            num.reshape(-1)[j] = (fp - fm) / (2 * eps)
        assert torch.allclose(x.grad, num, rtol=1e-4, atol=1e-6), (i, (x.grad - num).abs().max())


def test_register_search_recovers_shifts():
    """Exhaustive registration on a burst made with SatSynthBurst's recipe from a random texture."""
    from scipy.ndimage import gaussian_filter, shift as nd_shift
    from gsmisr.misr import register_search
    rng = np.random.default_rng(5)
    df, n = 4, 6
    hr = gaussian_filter(rng.uniform(0, 1, (200, 200, 3)), (2.0, 2.0, 0))
    hr = (hr - hr.min()) / (hr.max() - hr.min())
    d = np.r_[[[0.0, 0.0]], rng.uniform(-1, 1, (n - 1, 2))]          # LR px, generator convention
    frames = []
    for t in range(n):
        mv = nd_shift(hr, (-d[t, 1] * df, -d[t, 0] * df, 0), order=1, mode="nearest")[36:164, 36:164]
        x = gaussian_filter(mv, (1.0 / df, 1.0 / df, 0))
        frames.append(x.reshape(32, df, 32, df, 3).mean((1, 3)))
    b = register_search(np.stack(frames).astype(np.float32))
    err = np.abs(-b[1:] - d[1:]).max()
    assert err < 0.06, err


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
