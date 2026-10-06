#!/usr/bin/env python3
"""Checks for the scorer. numpy only:  python eval/tests/test_colormatch.py

1. DBSR colour matching removes a known colour cast.
2. The PSNR helper matches 10*log10(1/MSE).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import match_colors, psnr  # noqa: E402

rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:128, 0:128] / 128.0
gt = np.stack([0.5 + 0.3 * np.sin(6 * xx), 0.4 + 0.3 * np.cos(5 * yy), 0.3 + 0.2 * xx * yy], -1)
gt = np.clip(gt + 0.02 * rng.standard_normal(gt.shape), 0, 1).astype(np.float32)
pred = (gt @ np.array([[1.10, 0.03, 0.0], [0.0, 0.92, 0.02], [0.01, 0.0, 1.05]])).astype(np.float32)

p0 = psnr(pred, gt)
p_dbsr = psnr(match_colors(gt, pred, "dbsr"), gt)
print(f"no matching      {p0:6.2f} dB")
print(f"dbsr matching    {p_dbsr:6.2f} dB")
assert p_dbsr > p0 + 10, "DBSR colour matching should remove the cast"
assert abs(psnr(gt * 0 + 0.5, gt * 0 + 0.6) - 20.0) < 1e-4
print("ok")
