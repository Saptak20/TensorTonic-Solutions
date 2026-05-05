import numpy as np

def dropout(x, p=0.5, rng=None):
    if rng is None:
        rng = np.random

    x = np.array(x)  # 🔥 FIX: ensure numpy array

    mask = (rng.random(x.shape) > p).astype(x.dtype)
    scale = 1.0 / (1 - p)
    dropout_pattern = mask * scale
    output = x * dropout_pattern

    return output, dropout_pattern