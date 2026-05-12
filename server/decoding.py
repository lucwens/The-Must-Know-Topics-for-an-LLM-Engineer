"""Decoding strategies applied to real model logits — greedy, top-k, top-p,
plus a temperature knob. Returns the candidate distribution as well as which
token a given strategy would emit, so the UI can show 'this is the slice the
strategy is sampling from'."""
from __future__ import annotations

from typing import List

import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = x / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top_n_distribution(
    logits: np.ndarray,
    temperature: float,
    n_show: int = 20,
) -> List[dict]:
    """Return the top-`n_show` tokens after temperature scaling, with both
    raw logits and softmax probabilities."""
    probs = softmax(logits, temperature)
    idx = np.argsort(-probs)[:n_show]
    return [
        {"token_id": int(i), "logit": float(logits[i]), "prob": float(probs[i])}
        for i in idx
    ]


def apply_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= probs.shape[0]:
        return probs.copy()
    out = np.zeros_like(probs)
    idx = np.argpartition(-probs, k - 1)[:k]
    out[idx] = probs[idx]
    s = out.sum()
    return out / s if s > 0 else out


def apply_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return probs.copy()
    if p <= 0.0:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p) + 1
    keep = order[:cutoff]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    s = out.sum()
    return out / s if s > 0 else out


def strategy_pick(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    rng_seed: int,
) -> dict:
    """Apply each strategy and report which token id it would select. We use a
    fixed seed so 'sample again' shows deterministic-but-different draws when
    the user changes the seed."""
    rng = np.random.default_rng(int(rng_seed))

    base = softmax(logits, temperature)
    greedy_id = int(np.argmax(logits))

    after_k = apply_top_k(base, top_k)
    if after_k.sum() > 0:
        top_k_id = int(rng.choice(len(after_k), p=after_k))
    else:
        top_k_id = greedy_id

    after_p = apply_top_p(base, top_p)
    if after_p.sum() > 0:
        top_p_id = int(rng.choice(len(after_p), p=after_p))
    else:
        top_p_id = greedy_id

    return {
        "greedy": greedy_id,
        "top_k": top_k_id,
        "top_p": top_p_id,
        "top_k_kept": int((after_k > 0).sum()),
        "top_p_kept": int((after_p > 0).sum()),
    }
