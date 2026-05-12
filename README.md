# LLM Principles Explorer

An interactive web app that visualises the core building blocks described in
*The Must-Know Topics for an LLM Engineer* (Aliaksei Mikhailiuk, Towards Data
Science). Every numeric value the UI shows is computed from a real
[DistilGPT-2](https://huggingface.co/distilgpt2) forward pass — there are no
synthetic or hand-drawn numbers.

## What it covers

| Section | Real computation it performs |
| --- | --- |
| **1. Tokenization** | Runs the GPT-2 BPE tokenizer, shows token IDs and byte offsets. |
| **2. Embeddings** | Looks up rows of DistilGPT-2's `wte` matrix; computes cosine similarities and a 2-D PCA projection across the words you enter. Also lets you run a `a − b + c ≈ ?` analogy and see the honest cosine scores. |
| **3. Positional Encoding** | Computes the sinusoidal encoding from the original *Attention Is All You Need* paper, plus a RoPE rotation demo where you watch any vector rotate as the position index increases. |
| **4. Attention (Q · K · V)** | Hooks into `transformer.h[layer].attn.c_attn` to capture the **actual Q, K, V projections**, then walks you through the five steps: tokenize → project → score → mask + softmax → weighted sum of V. You can pick layer, head, and inspect any token's vectors. |
| **5. Decoding strategies** | Pulls the real next-token logits, applies temperature, then shows what greedy / top-k / top-p would each emit. A second panel runs an autoregressive loop and shows each emitted token together with the probability the model assigned to it. |
| **6. Architectures** | Static reference card for encoder-only / decoder-only / encoder-decoder. |

Topics from the article that don't lend themselves to live computation in a
single small model (pre-training, RLHF, RAG, FlashAttention, quantisation,
prompt engineering, evaluation) are described in the article itself — adding a
fake interactive visualisation for them would conflict with the
"no visual fakes" requirement, so they're acknowledged in the Architectures
tab rather than mocked up.

## Run it

Python 3.10+ recommended.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Or just `./run.sh`.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Or just `.\run.ps1`.

> If PowerShell refuses to run the activation script, run
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> once. Alternatively skip activation and call the venv's Python directly:
> `.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000`.

Then open <http://localhost:8000>. On first start the server downloads
DistilGPT-2 from the HuggingFace Hub (~330 MB) and then serves all
subsequent requests from memory.

## File layout

```
server/
  main.py        # FastAPI app, endpoints
  model.py       # DistilGPT-2 loading + Q/K/V extraction via forward hook
  positional.py  # sinusoidal + RoPE math
  decoding.py    # greedy / top-k / top-p / temperature
static/
  index.html     # one-page app with tabs per principle
  app.js         # frontend logic + Plotly plots
  style.css      # dark UI
```

## A note on honesty

GPT-2's `king − man + woman ≈ ?` analogy doesn't recover `queen` as cleanly as
word2vec does — the input embeddings of a contextual language model don't
sit in the same kind of nicely structured space. The analogy demo reports the
raw cosine similarities so you can see exactly where it works and where it
breaks down. Same principle applies everywhere else: the app shows the
numbers the model actually produces.
