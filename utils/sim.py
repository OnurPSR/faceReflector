from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SimParams:
    k_dst: float = 0.020
    damp: float = 0.97
    max_v: float = 2.0
    repel_radius: float = 0.95
    repel_strength: float = 0.06
    align_strength: float = 0.03
    dt: float = 1.0


def sim_step(pos: np.ndarray,
             vel: np.ndarray,
             dst: np.ndarray,
             sidelen: int,
             p: SimParams) -> tuple[np.ndarray, np.ndarray]:
    print("Performing simulation step...")

    d = dst - pos
    dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    acc = p.k_dst * d * dist / float(sidelen)

    H = W = sidelen
    N = pos.shape[0]
    cell_y = np.clip(pos[:, 0].astype(np.int32), 0, H - 1)
    cell_x = np.clip(pos[:, 1].astype(np.int32), 0, W - 1)
    # buckets: list of lists (simple; OK for sidelen<=128). For speed, could use arrays.
    buckets = [[] for _ in range(N)]
    for i in range(N):
        buckets[cell_y[i] * W + cell_x[i]].append(i)

    def neighbors_of(i):
        cy, cx = cell_y[i], cell_x[i]
        for ny in range(max(0, cy - 1), min(H, cy + 2)):
            for nx in range(max(0, cx - 1), min(W, cx + 2)):
                for j in buckets[ny * W + nx]:
                    if j != i:
                        yield j

    for i in range(N):
        # repulsion + alignment from nearby seeds
        v_sum = np.zeros((2,), dtype=np.float32)
        w_sum = 0.0
        for j in neighbors_of(i):
            dp = pos[i] - pos[j]
            d2 = float(dp[0]*dp[0] + dp[1]*dp[1]) + 1e-6
            d = math.sqrt(d2)
            if d < p.repel_radius:
                # repulse stronger when closer
                wj = (p.repel_radius - d) / p.repel_radius
                acc[i] += (dp / d) * (p.repel_strength * wj)
            # velocity alignment (soft)
            wv = 1.0 / (1.0 + d2)
            v_sum += vel[j] * wv
            w_sum += wv
        if w_sum > 0:
            v_avg = v_sum / w_sum
            acc[i] += (v_avg - vel[i]) * p.align_strength

    # Integrate with damping and clamp speed
    vel = vel + acc * p.dt
    vel = vel * p.damp
    speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8
    vel = vel * np.minimum(1.0, p.max_v / speed)
    pos = pos + vel * p.dt

    # Wall clamp
    pos[:, 0] = np.clip(pos[:, 0], 0, sidelen - 1)
    pos[:, 1] = np.clip(pos[:, 1], 0, sidelen - 1)
    return pos, vel
