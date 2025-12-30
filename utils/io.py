from __future__ import annotations

import numpy as np
from PIL import Image

def load_and_square_crop(path: str, sidelen: int) -> np.ndarray:

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((sidelen, sidelen), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr
