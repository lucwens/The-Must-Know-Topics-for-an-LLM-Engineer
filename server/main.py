"""FastAPI app — wires up the visualisation endpoints against a real
DistilGPT-2 instance. Static frontend is served from /."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA

from . import decoding, model, positional

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Loading DistilGPT-2 ...")
    bundle = model.load()
    print(
        f"Model ready: {model.MODEL_NAME} "
        f"(n_layer={bundle.n_layer}, n_head={bundle.n_head}, n_embd={bundle.n_embd})"
    )
    yield


app = FastAPI(title="LLM Principles Explorer", lifespan=lifespan)


@app.get("/api/info")
def info() -> dict:
    b = model.load()
    return {
        "model": model.MODEL_NAME,
        "n_layer": b.n_layer,
        "n_head": b.n_head,
        "n_embd": b.n_embd,
        "d_head": b.d_head,
        "vocab_size": b.tokenizer.vocab_size,
        "device": str(b.device),
    }


class TokenizeRequest(BaseModel):
    text: str = Field(..., max_length=2000)


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest) -> dict:
    tokens = model.tokenize(req.text)
    return {"tokens": tokens, "count": len(tokens)}


class EmbeddingsRequest(BaseModel):
    words: List[str] = Field(..., min_length=1, max_length=32)


@app.post("/api/embeddings")
def embeddings(req: EmbeddingsRequest) -> dict:
    """For each input word, return its first-token embedding vector plus a
    pairwise cosine-similarity matrix and a 2-D PCA projection so the user can
    see semantic clustering."""
    b = model.load()
    items = []
    vectors = []
    for w in req.words:
        # Prefix with a space — GPT-2 BPE treats leading space as part of the token.
        ids = b.tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = b.tokenizer(w, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        vec = model.token_embeddings([ids[0]])[0]
        items.append({
            "word": w,
            "token_id": int(ids[0]),
            "token_text": b.tokenizer.decode([ids[0]]),
            "n_subwords": len(ids),
            "preview": [float(v) for v in vec[:8]],
        })
        vectors.append(vec)
    if not vectors:
        raise HTTPException(status_code=400, detail="No valid tokens produced.")

    V = np.stack(vectors)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    Vn = V / norms
    sim = (Vn @ Vn.T).tolist()

    if len(vectors) >= 2:
        n_components = min(2, V.shape[0], V.shape[1])
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(V)
        if proj.shape[1] == 1:
            proj = np.concatenate([proj, np.zeros_like(proj)], axis=1)
    else:
        proj = np.zeros((1, 2))

    return {
        "items": items,
        "similarity": sim,
        "projection": proj.tolist(),
    }


class AnalogyRequest(BaseModel):
    a: str
    b: str
    c: str
    candidates: List[str] = Field(default_factory=list, max_length=16)


@app.post("/api/embedding-analogy")
def embedding_analogy(req: AnalogyRequest) -> dict:
    """Classic 'a - b + c ≈ ?' test using GPT-2's input embeddings. Real model
    weights — we report the cosine similarity to each candidate honestly, so
    you can see when the analogy works and when it doesn't."""
    tok = model.load().tokenizer

    def vec(word: str) -> np.ndarray:
        ids = tok(" " + word, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tok(word, add_special_tokens=False)["input_ids"]
        if not ids:
            raise HTTPException(status_code=400, detail=f"Could not tokenize '{word}'.")
        return model.token_embeddings([ids[0]])[0]

    target = vec(req.a) - vec(req.b) + vec(req.c)
    scored = []
    for w in req.candidates:
        try:
            v = vec(w)
            cos = float(np.dot(target, v) / (np.linalg.norm(target) * np.linalg.norm(v) + 1e-12))
            scored.append({"word": w, "cosine": cos})
        except HTTPException:
            continue
    scored.sort(key=lambda x: -x["cosine"])
    return {"expression": f"{req.a} - {req.b} + {req.c}", "results": scored}


class PositionalRequest(BaseModel):
    seq_len: int = Field(20, ge=1, le=128)
    d_model: int = Field(64, ge=2, le=256)


@app.post("/api/positional")
def positional_endpoint(req: PositionalRequest) -> dict:
    enc = positional.sinusoidal_encoding(req.seq_len, req.d_model)
    return {
        "seq_len": int(enc.shape[0]),
        "d_model": int(enc.shape[1]),
        "encoding": enc.tolist(),
    }


class RopeRequest(BaseModel):
    vector: List[float]
    positions: List[int]


@app.post("/api/rope")
def rope_endpoint(req: RopeRequest) -> dict:
    if len(req.vector) % 2 == 1:
        raise HTTPException(status_code=400, detail="Vector dimension must be even for RoPE.")
    vec = np.asarray(req.vector, dtype=np.float64)
    out = [positional.rope_rotate(vec, p).tolist() for p in req.positions]
    return {"positions": req.positions, "rotated": out}


class AttentionRequest(BaseModel):
    text: str = Field(..., max_length=400)
    layer: int = Field(0, ge=0)


@app.post("/api/attention")
def attention_endpoint(req: AttentionRequest) -> JSONResponse:
    b = model.load()
    if req.layer >= b.n_layer:
        raise HTTPException(status_code=400, detail=f"layer must be < {b.n_layer}")
    try:
        result = model.attention_pass(req.text, req.layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(result)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=400)
    temperature: float = Field(1.0, ge=0.05, le=5.0)
    top_k: int = Field(50, ge=0, le=1000)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    n_show: int = Field(20, ge=5, le=50)
    seed: int = Field(0, ge=0, le=2**31 - 1)


@app.post("/api/generate")
def generate_endpoint(req: GenerateRequest) -> dict:
    logits, prompt_tokens, _ = model.next_token_logits(req.prompt)
    top = decoding.top_n_distribution(logits, req.temperature, req.n_show)
    for entry in top:
        entry["text"] = model.decode_token(entry["token_id"])
    pick = decoding.strategy_pick(logits, req.temperature, req.top_k, req.top_p, req.seed)
    return {
        "prompt_tokens": prompt_tokens,
        "top": top,
        "picks": {
            **pick,
            "greedy_text": model.decode_token(pick["greedy"]),
            "top_k_text": model.decode_token(pick["top_k"]),
            "top_p_text": model.decode_token(pick["top_p"]),
        },
    }


class GenerateSeqRequest(BaseModel):
    prompt: str = Field(..., max_length=400)
    temperature: float = Field(1.0, ge=0.05, le=5.0)
    top_k: int = Field(50, ge=0, le=1000)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(20, ge=1, le=60)
    strategy: str = Field("top_p", pattern=r"^(greedy|top_k|top_p|sample)$")
    seed: int = Field(0, ge=0, le=2**31 - 1)


@app.post("/api/generate-sequence")
def generate_sequence(req: GenerateSeqRequest) -> dict:
    """Autoregressive generation with the chosen strategy. Each step uses real
    logits from DistilGPT-2; we just apply the user-selected filter."""
    rng = np.random.default_rng(int(req.seed))
    b = model.load()
    ids = b.tokenizer(req.prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [b.tokenizer.eos_token_id]
    steps = []
    cur = list(ids)
    for _ in range(req.max_new_tokens):
        logits, _, _ = model.next_token_logits(b.tokenizer.decode(cur))
        probs = decoding.softmax(logits, req.temperature)
        if req.strategy == "greedy":
            tok_id = int(np.argmax(logits))
        elif req.strategy == "top_k":
            filtered = decoding.apply_top_k(probs, req.top_k)
            tok_id = int(rng.choice(len(filtered), p=filtered))
        elif req.strategy == "top_p":
            filtered = decoding.apply_top_p(probs, req.top_p)
            tok_id = int(rng.choice(len(filtered), p=filtered))
        else:  # plain sample
            tok_id = int(rng.choice(len(probs), p=probs))
        cur.append(tok_id)
        steps.append({
            "token_id": tok_id,
            "text": b.tokenizer.decode([tok_id]),
            "prob": float(probs[tok_id]),
        })
        if tok_id == b.tokenizer.eos_token_id:
            break
    return {
        "completion": b.tokenizer.decode(cur),
        "new_text": b.tokenizer.decode(cur[len(ids):]),
        "steps": steps,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
