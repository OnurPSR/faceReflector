from __future__ import annotations

import numpy as np

def rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:

    a = 0.055
    rgb = np.clip(rgb, 0.0, 1.0)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    # f(t)
    eps = 216 / 24389  # (6/29)^3
    kappa = 24389 / 27
    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a_ = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return np.stack([L, a_, b_], axis=-1).astype(np.float32)
