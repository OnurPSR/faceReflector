from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:
    torch = None


def shift_no_wrap(arr: "torch.Tensor", dy: int, dx: int, fill: int) -> "torch.Tensor":
    print("Shifting tensor without wrap-around...")

    H, W = arr.shape
    out = torch.full_like(arr, fill)
    y0 = max(0, dy)
    y1 = H + min(0, dy)
    x0 = max(0, dx)
    x1 = W + min(0, dx)
    out[y0:y1, x0:x1] = arr[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def jfa_voronoi_torch(seed_yx: "torch.Tensor", H: int, W: int) -> "torch.Tensor":
    print("Running JFA Voronoi on torch...")

    device = seed_yx.device
    N = seed_yx.shape[0]

    ids = torch.full((H, W), -1, dtype=torch.int64, device=device)

    ys = torch.clamp(seed_yx[:, 0].round().to(torch.int64), 0, H - 1)
    xs = torch.clamp(seed_yx[:, 1].round().to(torch.int64), 0, W - 1)
    ids[ys, xs] = torch.arange(N, device=device, dtype=torch.int64)

    yy = torch.arange(H, device=device).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device).view(1, W).expand(H, W)

    best_id = ids.clone()
    best_d2 = torch.full((H, W), float("inf"), device=device)

    # initialize best_d2 where ids are set
    valid = best_id >= 0
    if valid.any():
        cand = best_id[valid]
        cy = seed_yx[cand, 0]
        cx = seed_yx[cand, 1]
        best_d2[valid] = (yy[valid].float() - cy) ** 2 + (xx[valid].float() - cx) ** 2

    step = 1
    while step < max(H, W):
        step *= 2
    step //= 2

    while step >= 1:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                cand_id = best_id if (dy == 0 and dx == 0) else shift_no_wrap(best_id, dy, dx, -1)
                valid = cand_id >= 0
                if not valid.any():
                    continue
                safe = cand_id.clamp(min=0)
                cy = seed_yx[safe, 0]
                cx = seed_yx[safe, 1]
                d2 = (yy.float() - cy) ** 2 + (xx.float() - cx) ** 2
                d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
                better = d2 < best_d2
                best_d2 = torch.where(better, d2, best_d2)
                best_id = torch.where(better, cand_id, best_id)
        step //= 2

    return best_id


def render_frame(seed_pos_yx: np.ndarray,
                 seed_rgb: np.ndarray,
                 sidelen: int,
                 device: str = "cpu") -> np.ndarray:
    print(f"Rendering frame on device: {device}")

    if torch is None:
        raise RuntimeError("PyTorch is required for fast rendering (install torch).")

    dev = torch.device(device)
    pos = torch.from_numpy(seed_pos_yx.astype(np.float32)).to(dev)
    col = torch.from_numpy(seed_rgb.astype(np.float32)).to(dev)  # (N,3)

    ids = jfa_voronoi_torch(pos, sidelen, sidelen)
    valid = ids >= 0
    out = torch.zeros((sidelen, sidelen, 3), device=dev, dtype=torch.float32)
    out[valid] = col[ids[valid]]
    out = (out.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    return out
