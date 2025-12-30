from __future__ import annotations

import numpy as np

def importance_from_edges(rgb: np.ndarray, alpha: float = 2.0) -> np.ndarray:

    gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    # Sobel filters
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    # convolution (small, so simple padding + sliding)
    pad = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            gx += Kx[dy, dx] * pad[dy:dy+gray.shape[0], dx:dx+gray.shape[1]]
            gy += Ky[dy, dx] * pad[dy:dy+gray.shape[0], dx:dx+gray.shape[1]]

    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)
    w = 1.0 + alpha * mag
    return w.astype(np.float32)
