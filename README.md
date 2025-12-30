# Pixel Flow Mosaic (Python)

This project takes a **source image** and **rearranges its pixels** **without changing any RGB values** so that the final result **matches the *structure* of a target image**.

This is a “pixel-identity-preserving” transform.

## What the algorithm is doing (high-level)

The system splits the problem into two stages:

1. **Decide the final layout (assignment / permutation).**  
   Solve a global “who-goes-where” problem: for each **target position**, pick exactly one **source pixel** to occupy it. This produces a **one-to-one permutation** of the source pixels.

2. **Turn that layout into motion + visuals.**  
   Once every pixel has a destination, simulate a smooth movement over time. Each frame is rendered by treating pixels as moving **seeds** and constructing a Voronoi mosaic using the **Jump Flood Algorithm (JFA)** for efficiency.

---

### 0) Inputs & configuration
- Read **source** and **target** images.
- Parse parameters (`sidelen`, `iters`, `proximity`, `edge_alpha`, `frames`, etc.).

### 1) Preprocess images (make the permutation possible)
1. **Square-crop** both images to a shared aspect ratio.
2. **Resize** both to `sidelen × sidelen` so they have the *same number of pixels*.
3. Convert to arrays:
   - Keep **RGB** for final rendering (must be preserved).
   - Convert to **Lab** (perceptual color space) for comparing colors during assignment.
4. Build coordinate grids:
   - For each pixel index, store its (x, y) location. This enables spatial penalties later.

### 2) Build target structure weights (edge/contrast guidance)
1. Compute an edge-strength (contrast) map on the **target**.
2. Convert it into per-pixel weights:
   - High weights on strong boundaries.
   - Low weights on flat regions.
3. `edge_alpha` controls how strongly these edge weights influence assignment decisions.

### 3) Compute the assignment permutation (target positions → source pixels)

**Goal:** for every target pixel position `t`, choose a unique source pixel `s` minimizing a combined cost with three intuitive parts:

- **Color mismatch (Lab distance):**  
  prefer source pixels whose perceived color is close to what the target “wants” at that location.

- **Edge emphasis (target weights):**  
  enforce better matches on important structural pixels (high-edge areas).

- **Spatial regularization (distance penalty):**  
  discourage assignments that require very long moves. `proximity` sets how strongly long-distance travel is penalized.

**How it’s typically solved:** a **swap-search heuristic** instead of an exact (expensive) matching solver:
1. Start with an initial assignment (random / identity / cheap greedy).
2. Repeatedly propose swaps in the permutation.
3. Accept swaps that improve the objective.
4. Run for `iters` iterations.

### 4) Invert the mapping (source pixels → destination coordinates)
Assignment is naturally expressed as:

- **target position → source pixel**

But motion simulation needs:

- **source pixel → destination position**

So we invert the mapping to obtain a destination (x, y) for every source pixel.

### 5) Simulate smooth pixel flow (moving seeds over frames)
Treat each source pixel as a **seed** with:
- a fixed **color** (its RGB),
- a time-varying **position** moving from start → destination.

Output: for each frame `k`, a full set of seed positions `P_k` (one position per pixel).

### 6) Render each frame as a Voronoi mosaic (JFA)
For each frame:
1. Consider the seed set `P_k`.
2. For every output pixel coordinate, find the **nearest seed**.
3. Color the output pixel with that seed’s **source RGB**.

This constructs a Voronoi partition (cells closest to each seed).

### 7) Export animation
- Assemble frames in order.
- Save as GIF.

---

## Output examples

### Source
![Example source](examples/source.jpg)

### Target
![Example target](examples/target.jpg)

### Output GIF (illustrative)
![Example output](examples/out.gif)

> The examples are generated from synthetic images in `examples/`.  
> The real outputs depend on `sidelen`, `iters`, and how detailed the target is.

---

## Installation

Python 3.10+ recommended.

```bash
pip install numpy pillow imageio torch
```

## Run

```bash
python main.py \
  --source path/to/source.jpg \
  --target path/to/target.jpg \
  --sidelen 96 \
  --frames 120 \
  --iters 800000 \
  --proximity 0.025 \
  --edge_alpha 4.0 \
  --out out.gif
```

---

## Parameter intuition

- `--sidelen` : Resolution. Higher = better, slower (start 64 → 96 → 128).  
- `--iters` : Swap-search iterations for the permutation. Higher = better match, slower.  
- `--proximity` : Discourages long pixel moves. Too high can “freeze” the morph.  
- `--edge_alpha` : Emphasizes target edges. Too high can look noisy.  
- `--frames` : Animation length (does not change the computed assignment).

