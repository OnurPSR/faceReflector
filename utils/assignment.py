from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def initial_perm_sort(src_lab: np.ndarray, tgt_lab: np.ndarray) -> np.ndarray:

    H, W, _ = src_lab.shape
    N = H * W
    s = src_lab.reshape(N, 3)
    t = tgt_lab.reshape(N, 3)

    # Key: mostly luminance, slight chroma
    s_key = 0.70 * s[:, 0] + 0.15 * s[:, 1] + 0.15 * s[:, 2]
    t_key = 0.70 * t[:, 0] + 0.15 * t[:, 1] + 0.15 * t[:, 2]
    s_order = np.argsort(s_key)
    t_order = np.argsort(t_key)

    perm = np.empty(N, dtype=np.int32)
    perm[t_order] = s_order
    return perm


def cost_at(dst_idx: int,
            src_idx: int,
            tgt_lab_flat: np.ndarray,
            src_lab_flat: np.ndarray,
            w_flat: np.ndarray,
            dst_y: np.ndarray, dst_x: np.ndarray,
            src_y: np.ndarray, src_x: np.ndarray,
            proximity_importance: float) -> float:

    # Color term (Lab distance), weighted by importance at target.
    d = tgt_lab_flat[dst_idx] - src_lab_flat[src_idx]
    color = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]) * w_flat[dst_idx]

    dy = float(dst_y[dst_idx] - src_y[src_idx])
    dx = float(dst_x[dst_idx] - src_x[src_idx])
    dist2 = dy*dy + dx*dx
    prox = proximity_importance * dist2

    prox = prox * prox

    return float(color + prox)


def refine_perm_swaps(perm: np.ndarray,
                      src_lab: np.ndarray,
                      tgt_lab: np.ndarray,
                      w: np.ndarray,
                      proximity_importance: float,
                      iters: int = 200_000,
                      start_radius: int | None = None,
                      seed: int = 0,
                      anneal: bool = True) -> np.ndarray:

    rng = np.random.default_rng(seed)
    H, W, _ = src_lab.shape
    N = H * W

    src_lab_flat = src_lab.reshape(N, 3)
    tgt_lab_flat = tgt_lab.reshape(N, 3)
    w_flat = w.reshape(N)

    dst_y, dst_x = np.divmod(np.arange(N, dtype=np.int32), W)
    src_y, src_x = np.divmod(np.arange(N, dtype=np.int32), W)

    if start_radius is None:
        start_radius = max(4, H // 2)

    T0 = 1.0
    T1 = 0.02

    cur_cost = np.zeros(N, dtype=np.float64)
    for j in range(N):
        cur_cost[j] = cost_at(j, int(perm[j]), tgt_lab_flat, src_lab_flat, w_flat,
                              dst_y, dst_x, src_y, src_x, proximity_importance)

    radius = start_radius
    stage_iters = max(10_000, iters // int(math.log2(radius) + 1))

    accepted = 0
    for it in range(iters):
        if it > 0 and it % stage_iters == 0 and radius > 1:
            radius = max(1, radius // 2)

        a = int(rng.integers(0, N))
        ay, ax = int(dst_y[a]), int(dst_x[a])
        by = int(np.clip(ay + rng.integers(-radius, radius + 1), 0, H - 1))
        bx = int(np.clip(ax + rng.integers(-radius, radius + 1), 0, W - 1))
        b = by * W + bx
        if a == b:
            continue

        src_a = int(perm[a])
        src_b = int(perm[b])

        new_cost_a = cost_at(a, src_b, tgt_lab_flat, src_lab_flat, w_flat,
                             dst_y, dst_x, src_y, src_x, proximity_importance)
        new_cost_b = cost_at(b, src_a, tgt_lab_flat, src_lab_flat, w_flat,
                             dst_y, dst_x, src_y, src_x, proximity_importance)

        old_cost = cur_cost[a] + cur_cost[b]
        new_cost = new_cost_a + new_cost_b
        delta = new_cost - old_cost

        accept = delta < 0
        if (not accept) and anneal:
            t = it / max(1, iters - 1)
            T = (1 - t) * T0 + t * T1

            if delta > 0 and rng.random() < math.exp(-delta / max(1e-9, T)):
                accept = True

        if accept:
            perm[a], perm[b] = perm[b], perm[a]
            cur_cost[a] = new_cost_a
            cur_cost[b] = new_cost_b
            accepted += 1

    return perm


def dest_for_each_src_from_perm(perm: np.ndarray, H: int, W: int) -> np.ndarray:

    N = H * W
    dst_y, dst_x = np.divmod(np.arange(N, dtype=np.int32), W)
    inv = np.empty(N, dtype=np.int32)
    inv[perm] = np.arange(N, dtype=np.int32)
    dst_of_src = np.stack([dst_y[inv], dst_x[inv]], axis=1).astype(np.float32)
    return dst_of_src


def load_perm_json(path: str | Path, N: int) -> np.ndarray:

    p = Path(path)
    obj: Any = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        if len(obj) != N:
            raise ValueError(f"JSON list length {len(obj)} != N={N}")
        perm = np.asarray(obj, dtype=np.int32)
    elif isinstance(obj, dict):
        perm = np.full((N,), -1, dtype=np.int32)
        for k, v in obj.items():
            dk = int(k)
            perm[dk] = int(v)
        if (perm < 0).any():
            raise ValueError("JSON dict did not specify all destinations 0..N-1")
    else:
        raise TypeError(f"Unsupported JSON structure: {type(obj)}")
    return perm
