"""Sinusoidal positional encoding from the original Transformer paper.
Real arithmetic, no faked values — useful for the heatmap view."""
from __future__ import annotations

import numpy as np


def sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    seq_len = max(1, min(int(seq_len), 128))
    d_model = max(2, min(int(d_model), 256))
    if d_model % 2 == 1:
        d_model += 1
    pos = np.arange(seq_len, dtype=np.float64)[:, None]
    i = np.arange(d_model // 2, dtype=np.float64)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * i) / d_model)
    angles = pos * angle_rates  # (seq_len, d_model/2)
    out = np.zeros((seq_len, d_model), dtype=np.float64)
    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles)
    return out


def rope_rotate(vec: np.ndarray, position: int, base: float = 10000.0) -> np.ndarray:
    """Apply RoPE rotation to a single vector at a given position.
    Treats the vector as a sequence of 2-d pairs and rotates each pair by an
    angle derived from its dimension index."""
    vec = np.asarray(vec, dtype=np.float64).copy()
    d = vec.shape[0]
    if d % 2 == 1:
        raise ValueError("RoPE requires an even-dimensional vector.")
    half = d // 2
    freqs = 1.0 / np.power(base, np.arange(half, dtype=np.float64) * 2 / d)
    angles = position * freqs
    cos = np.cos(angles)
    sin = np.sin(angles)
    even = vec[0::2]
    odd = vec[1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    out = np.empty_like(vec)
    out[0::2] = rotated_even
    out[1::2] = rotated_odd
    return out
