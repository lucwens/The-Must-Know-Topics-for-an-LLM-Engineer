"""Loads DistilGPT-2 once and exposes helpers for tokenization, embeddings,
Q/K/V extraction, and next-token logits. All values returned are real numbers
computed by the actual model — no synthetic data."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "distilgpt2"


@dataclass
class ModelBundle:
    tokenizer: GPT2TokenizerFast
    model: GPT2LMHeadModel
    device: torch.device

    @property
    def n_layer(self) -> int:
        return self.model.config.n_layer

    @property
    def n_head(self) -> int:
        return self.model.config.n_head

    @property
    def n_embd(self) -> int:
        return self.model.config.n_embd

    @property
    def d_head(self) -> int:
        return self.n_embd // self.n_head


_bundle: ModelBundle | None = None


def load() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _bundle = ModelBundle(tokenizer=tokenizer, model=model, device=device)
    return _bundle


def tokenize(text: str) -> List[dict]:
    b = load()
    enc = b.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    out = []
    for i, (tid, (start, end)) in enumerate(zip(ids, offsets)):
        piece = b.tokenizer.decode([tid])
        raw = b.tokenizer.convert_ids_to_tokens(tid)
        out.append({
            "index": i,
            "id": int(tid),
            "text": piece,
            "raw": raw,
            "start": int(start),
            "end": int(end),
        })
    return out


def token_embeddings(token_ids: List[int]) -> np.ndarray:
    """Static (non-contextual) input embeddings — i.e. the wte lookup matrix.
    Shape: (T, n_embd)."""
    b = load()
    with torch.no_grad():
        wte = b.model.transformer.wte.weight  # (vocab, n_embd)
        ids = torch.tensor(token_ids, device=b.device)
        return wte[ids].detach().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def attention_pass(text: str, layer: int) -> dict:
    """Run a forward pass and capture Q, K, V at the requested layer plus the
    attention weights for every head. The captured tensors are the genuine
    model activations, not approximations."""
    b = load()
    enc = b.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) == 0:
        raise ValueError("Input produced zero tokens.")
    if len(ids) > 48:
        # Keep visualisations manageable.
        ids = ids[:48]

    input_ids = torch.tensor([ids], device=b.device)
    block = b.model.transformer.h[layer]
    captured: dict = {}

    def hook(_module, inputs, output):
        # GPT2Attention.c_attn projects hidden_state -> [Q | K | V] concatenated.
        x = inputs[0]  # (1, T, n_embd)
        qkv = output  # (1, T, 3 * n_embd)
        q, k, v = qkv.split(b.n_embd, dim=2)
        captured["hidden"] = x.detach().cpu().numpy()[0]
        captured["q"] = q.detach().cpu().numpy()[0]
        captured["k"] = k.detach().cpu().numpy()[0]
        captured["v"] = v.detach().cpu().numpy()[0]

    handle = block.attn.c_attn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = b.model(input_ids, output_attentions=True, use_cache=False)
    finally:
        handle.remove()

    attentions = out.attentions[layer][0].detach().cpu().numpy()  # (H, T, T)

    # Reshape Q, K, V into (H, T, d_head).
    T = len(ids)
    q = captured["q"].reshape(T, b.n_head, b.d_head).transpose(1, 0, 2)
    k = captured["k"].reshape(T, b.n_head, b.d_head).transpose(1, 0, 2)
    v = captured["v"].reshape(T, b.n_head, b.d_head).transpose(1, 0, 2)

    # Scores prior to mask + softmax: QK^T / sqrt(d_head). We compute this
    # ourselves so the UI can show "raw scores -> masked -> softmax" as three
    # stages — the model itself only exposes the final softmaxed weights.
    scale = 1.0 / math.sqrt(b.d_head)
    scores_pre = np.matmul(q, k.transpose(0, 2, 1)) * scale  # (H, T, T)
    # Causal mask (decoder-only model): future positions are -inf, then softmax.
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores_masked = np.where(causal[None], -np.inf, scores_pre)

    tokens = [
        {"index": i, "id": int(tid), "text": b.tokenizer.decode([tid]),
         "raw": b.tokenizer.convert_ids_to_tokens(tid)}
        for i, tid in enumerate(ids)
    ]
    return {
        "tokens": tokens,
        "n_head": b.n_head,
        "d_head": b.d_head,
        "n_embd": b.n_embd,
        "n_layer": b.n_layer,
        "layer": layer,
        "hidden": captured["hidden"].tolist(),
        "q": q.tolist(),
        "k": k.tolist(),
        "v": v.tolist(),
        "scores_pre": scores_pre.tolist(),
        "scores_masked": np.where(np.isneginf(scores_masked), None, scores_masked).tolist(),
        "attention": attentions.tolist(),
    }


def next_token_logits(prompt: str) -> Tuple[np.ndarray, list, list]:
    """Returns logits for the position after the last prompt token, plus the
    decoded token list."""
    b = load()
    ids = b.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        # Fall back to BOS so the model has something to condition on.
        ids = [b.tokenizer.eos_token_id]
    input_ids = torch.tensor([ids], device=b.device)
    with torch.no_grad():
        out = b.model(input_ids, use_cache=False)
    logits = out.logits[0, -1].detach().cpu().numpy()  # (vocab,)
    tokens = [
        {"index": i, "id": int(tid), "text": b.tokenizer.decode([tid])}
        for i, tid in enumerate(ids)
    ]
    return logits, tokens, ids


def decode_token(token_id: int) -> str:
    return load().tokenizer.decode([token_id])
