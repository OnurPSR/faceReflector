from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from utils.io import load_and_square_crop
from utils.color import rgb_to_lab_np
from utils.edges import importance_from_edges
from utils.assignment import (
    initial_perm_sort,
    refine_perm_swaps,
    dest_for_each_src_from_perm,
    load_perm_json,
)
from utils.sim import SimParams, sim_step
from utils.voronoi import render_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="selfies/onurnur.jpeg", help="Path to source image (your photo).")
    ap.add_argument("--target", default="target.jpg", help="Path to target image (e.g., Obama). Required if no --assignment_json.")
    ap.add_argument("--sidelen", type=int, default=128, help="Working resolution (e.g., 64/96/128). Larger = slower.")
    ap.add_argument("--frames", type=int, default=120, help="Number of animation frames.")
    ap.add_argument("--out", default="out.gif", help="Output gif path.")
    ap.add_argument("--assignment_json", default=None, help="Optional: presets/*/assignments.json (dest->src).")
    ap.add_argument("--iters", type=int, default=800_000, help="Swap iterations for assignment search.")
    ap.add_argument("--proximity", type=float, default=0.025, help="Spatial penalty strength.")
    ap.add_argument("--edge_alpha", type=float, default=4.0, help="Edge importance multiplier for the target.")
    ap.add_argument("--device", default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    args = ap.parse_args()
    print("Arguments:", args)
    if imageio is None:
        raise RuntimeError("imageio is required to write GIFs (pip install imageio).")

    sidelen = int(args.sidelen)
    src_rgb = load_and_square_crop(args.source, sidelen)
    H = W = sidelen
    N = H * W

    # seed colors fixed from source pixels
    seed_rgb = src_rgb.reshape(N, 3).copy()

    # seed initial positions on grid centers
    yy, xx = np.divmod(np.arange(N, dtype=np.int32), W)
    seed_pos = np.stack([yy, xx], axis=1).astype(np.float32)
    print("Source and target images loaded.")

    # --- Assignment stage (dest->src permutation) ---
    if args.assignment_json:
        perm = load_perm_json(args.assignment_json, N)
    else:
        tgt_rgb = load_and_square_crop(args.target, sidelen)
        src_lab = rgb_to_lab_np(src_rgb)
        tgt_lab = rgb_to_lab_np(tgt_rgb)
        w = importance_from_edges(tgt_rgb, alpha=float(args.edge_alpha))
        perm = initial_perm_sort(src_lab, tgt_lab)
        perm = refine_perm_swaps(
            perm,
            src_lab=src_lab,
            tgt_lab=tgt_lab,
            w=w,
            proximity_importance=float(args.proximity),
            iters=int(args.iters),
            start_radius=None,
            seed=0,
            anneal=True,
        )

    dst_of_src = dest_for_each_src_from_perm(perm, H, W).astype(np.float32)
    print("Assignment computed.")

    # --- Simulate + render ---
    pos = seed_pos.copy()
    vel = np.zeros_like(pos)

    sim = SimParams()
    frames = []
    for _ in range(int(args.frames)):
        frame = render_frame(pos, seed_rgb, sidelen, device=args.device)
        frames.append(frame)
        pos, vel = sim_step(pos, vel, dst_of_src, sidelen, sim)

    print("Simulation complete.")
    out_path = Path(args.out)
    imageio.mimsave(out_path.as_posix(), frames, duration=0.03)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
