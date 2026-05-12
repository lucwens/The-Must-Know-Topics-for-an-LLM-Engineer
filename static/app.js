// LLM Principles Explorer — frontend.
// Every numeric value rendered here is computed by the backend running a real
// DistilGPT-2 forward pass.

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e6edf3", family: "ui-monospace, Menlo, monospace", size: 11 },
  margin: { l: 70, r: 20, t: 20, b: 50 },
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

const ATTN_COLORSCALE = [
  [0, "#161b22"], [0.25, "#2c3e72"], [0.5, "#5e7bbd"],
  [0.75, "#a3b8e8"], [1, "#f7f7ff"],
];
const SIGNED_COLORSCALE = [
  [0, "#f7768e"], [0.5, "#161b22"], [1, "#7aa2f7"],
];

async function api(path, body) {
  const opts = { method: "POST", headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  if (!body) opts.method = "GET";
  const r = await fetch(path, opts);
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${r.status}: ${txt}`);
  }
  return r.json();
}

// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});
document.querySelectorAll(".flow-step").forEach((el) => {
  el.addEventListener("click", () => activateTab(el.dataset.jump));
});
function activateTab(name) {
  document.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.dataset.panel === name)
  );
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- Model info ----------
let MODEL = null;
(async () => {
  try {
    MODEL = await api("/api/info");
    document.getElementById("modelInfo").textContent =
      `${MODEL.model} · ${MODEL.n_layer}×${MODEL.n_head}h · d=${MODEL.n_embd} · ${MODEL.device}`;
    const layerSel = document.getElementById("attnLayer");
    for (let i = 0; i < MODEL.n_layer; i++) {
      const o = document.createElement("option");
      o.value = i; o.textContent = i;
      layerSel.appendChild(o);
    }
    const headSel = document.getElementById("attnHead");
    for (let i = 0; i < MODEL.n_head; i++) {
      const o = document.createElement("option");
      o.value = i; o.textContent = i;
      headSel.appendChild(o);
    }
  } catch (e) {
    document.getElementById("modelInfo").textContent = "model load failed";
    console.error(e);
  }
})();

// ---------- Tokenization ----------
document.getElementById("tokGo").addEventListener("click", runTokenize);
document.getElementById("tokInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runTokenize();
});
async function runTokenize() {
  const text = document.getElementById("tokInput").value;
  const out = document.getElementById("tokOutput");
  const stats = document.getElementById("tokStats");
  out.innerHTML = ""; stats.innerHTML = "";
  try {
    const data = await api("/api/tokenize", { text });
    data.tokens.forEach((t, i) => {
      const el = document.createElement("div");
      el.className = "token";
      el.style.animationDelay = `${i * 30}ms`;
      const visible = t.text.replace(/ /g, "·").replace(/\n/g, "↵") || "∅";
      el.innerHTML = `<span class="text">${escapeHtml(visible)}</span><span class="id">#${t.id}</span>`;
      out.appendChild(el);
    });
    const chars = text.length;
    const ratio = chars > 0 ? (chars / Math.max(1, data.count)).toFixed(2) : "—";
    stats.innerHTML = `
      <span class="label">tokens</span><span class="value">${data.count}</span>
      <span class="label">characters</span><span class="value">${chars}</span>
      <span class="label">chars/token</span><span class="value">${ratio}</span>
      <span class="label">vocab</span><span class="value">${MODEL?.vocab_size ?? "—"}</span>
    `;
  } catch (e) { alert(e.message); }
}
runTokenize();

// ---------- Embeddings ----------
document.getElementById("embGo").addEventListener("click", runEmbed);
async function runEmbed() {
  const raw = document.getElementById("embInput").value;
  const words = raw.split(",").map((s) => s.trim()).filter(Boolean);
  if (words.length === 0) return;
  try {
    const data = await api("/api/embeddings", { words });
    // PCA scatter
    Plotly.react(
      "embProj",
      [{
        type: "scatter",
        mode: "markers+text",
        x: data.projection.map((p) => p[0]),
        y: data.projection.map((p) => p[1]),
        text: data.items.map((it) => it.word),
        textposition: "top center",
        marker: { size: 12, color: "#7aa2f7", line: { color: "#bb9af7", width: 1 } },
        textfont: { color: "#e6edf3" },
      }],
      { ...PLOT_LAYOUT, xaxis: { gridcolor: "#2a3142" }, yaxis: { gridcolor: "#2a3142" } },
      PLOT_CONFIG
    );
    // Similarity heatmap
    Plotly.react(
      "embSim",
      [{
        type: "heatmap",
        z: data.similarity,
        x: data.items.map((it) => it.word),
        y: data.items.map((it) => it.word),
        colorscale: SIGNED_COLORSCALE,
        zmin: -1, zmax: 1,
      }],
      { ...PLOT_LAYOUT, xaxis: { tickangle: -45 } },
      PLOT_CONFIG
    );
    // Preview table
    const tbl = document.getElementById("embTable");
    tbl.innerHTML = "";
    data.items.forEach((it) => {
      const row = document.createElement("div");
      row.className = "emb-row";
      const subwordNote = it.n_subwords > 1 ? ` (${it.n_subwords} subwords, showing 1st)` : "";
      row.innerHTML = `
        <div class="word">${escapeHtml(it.word)}<span style="color:#9da7b3;font-size:0.75rem">${subwordNote}</span></div>
        <div class="vec">[${it.preview.map((v) => v.toFixed(3)).join(", ")}, …]</div>
      `;
      tbl.appendChild(row);
    });
  } catch (e) { alert(e.message); }
}
runEmbed();

document.getElementById("anGo").addEventListener("click", async () => {
  const a = document.getElementById("anA").value.trim();
  const b = document.getElementById("anB").value.trim();
  const c = document.getElementById("anC").value.trim();
  const cands = document.getElementById("anCands").value
    .split(",").map((s) => s.trim()).filter(Boolean);
  if (!a || !b || !c || cands.length === 0) return;
  try {
    const data = await api("/api/embedding-analogy", { a, b, c, candidates: cands });
    const out = document.getElementById("anOut");
    out.innerHTML = "";
    const maxAbs = Math.max(...data.results.map((r) => Math.abs(r.cosine)), 0.001);
    data.results.forEach((r, idx) => {
      const row = document.createElement("div");
      row.className = "an-row" + (idx === 0 ? " best" : "");
      const w = Math.max(2, (Math.abs(r.cosine) / maxAbs) * 200);
      row.innerHTML = `
        <span class="word">${escapeHtml(r.word)}</span>
        <span><span class="bar" style="display:inline-block;width:${w}px"></span></span>
        <span class="cos">${r.cosine.toFixed(3)}</span>
      `;
      out.appendChild(row);
    });
  } catch (e) { alert(e.message); }
});

// ---------- Positional encoding ----------
document.getElementById("posGo").addEventListener("click", runPositional);
async function runPositional() {
  const seq_len = +document.getElementById("posLen").value;
  const d_model = +document.getElementById("posD").value;
  try {
    const data = await api("/api/positional", { seq_len, d_model });
    Plotly.react(
      "posHeat",
      [{
        type: "heatmap",
        z: data.encoding,
        colorscale: SIGNED_COLORSCALE,
        zmin: -1, zmax: 1,
      }],
      { ...PLOT_LAYOUT,
        xaxis: { title: "dimension", gridcolor: "#2a3142" },
        yaxis: { title: "position", gridcolor: "#2a3142" } },
      PLOT_CONFIG
    );
    // Show a few dimension waves over position
    const lines = [];
    const dims = [0, 1, 4, 5, Math.floor(data.d_model / 4), Math.floor(data.d_model / 2)];
    dims.forEach((d) => {
      if (d >= data.d_model) return;
      lines.push({
        type: "scatter", mode: "lines",
        x: Array.from({ length: data.seq_len }, (_, i) => i),
        y: data.encoding.map((row) => row[d]),
        name: `dim ${d}`,
      });
    });
    Plotly.react("posLines", lines,
      { ...PLOT_LAYOUT,
        xaxis: { title: "position", gridcolor: "#2a3142" },
        yaxis: { title: "value", gridcolor: "#2a3142", range: [-1.1, 1.1] },
        legend: { font: { color: "#e6edf3" } } },
      PLOT_CONFIG);
  } catch (e) { alert(e.message); }
}
runPositional();

document.getElementById("ropeGo").addEventListener("click", runRope);
async function runRope() {
  const vec = document.getElementById("ropeVec").value
    .split(",").map((s) => parseFloat(s.trim())).filter((v) => !Number.isNaN(v));
  if (vec.length % 2 === 1) vec.push(0);
  if (vec.length === 0) return;
  const maxPos = Math.max(1, Math.min(32, +document.getElementById("ropeMax").value || 8));
  const positions = Array.from({ length: maxPos }, (_, i) => i);
  try {
    const data = await api("/api/rope", { vector: vec, positions });
    // Plot first 2 dimensions as a rotating arrow trail
    const traces = [];
    const xs = data.rotated.map((r) => r[0]);
    const ys = data.rotated.map((r) => r[1]);
    traces.push({
      type: "scatter", mode: "markers+lines",
      x: xs, y: ys, name: "dims [0,1]",
      marker: { size: 10, color: data.positions, colorscale: "Plasma", showscale: true,
                colorbar: { title: "position", titlefont: { color: "#e6edf3" }, tickfont: { color: "#e6edf3" } } },
      line: { color: "#7aa2f7", width: 1 },
    });
    if (vec.length >= 4) {
      const xs2 = data.rotated.map((r) => r[2]);
      const ys2 = data.rotated.map((r) => r[3]);
      traces.push({
        type: "scatter", mode: "markers+lines",
        x: xs2, y: ys2, name: "dims [2,3]",
        marker: { size: 8, color: data.positions, colorscale: "Viridis" },
        line: { color: "#bb9af7", width: 1, dash: "dot" },
      });
    }
    Plotly.react("ropePlot", traces, {
      ...PLOT_LAYOUT,
      xaxis: { title: "x", gridcolor: "#2a3142", scaleanchor: "y", scaleratio: 1 },
      yaxis: { title: "y", gridcolor: "#2a3142" },
      legend: { font: { color: "#e6edf3" } },
    }, PLOT_CONFIG);
  } catch (e) { alert(e.message); }
}
runRope();

// ---------- Attention ----------
let ATTN_STATE = null;
let ATTN_SELECTED_TOKEN = 0;
document.getElementById("attnGo").addEventListener("click", runAttention);
document.getElementById("attnHead").addEventListener("change", () => {
  if (ATTN_STATE) renderAttention();
});
document.getElementById("attnLayer").addEventListener("change", () => {
  // require re-running because Q/K/V live in a specific layer
  if (document.getElementById("attnInput").value) runAttention();
});

async function runAttention() {
  const text = document.getElementById("attnInput").value;
  const layer = +document.getElementById("attnLayer").value;
  try {
    ATTN_STATE = await api("/api/attention", { text, layer });
    ATTN_SELECTED_TOKEN = 0;
    renderAttention();
    revealSteps();
  } catch (e) { alert(e.message); }
}

function revealSteps() {
  const steps = document.querySelectorAll("#attnSteps .step");
  steps.forEach((s, i) => {
    s.classList.remove("revealed");
    setTimeout(() => s.classList.add("revealed"), 80 + i * 220);
  });
}

function renderAttention() {
  const head = +document.getElementById("attnHead").value;
  const s = ATTN_STATE;
  const labels = s.tokens.map((t) => visibleToken(t.text));

  // Step 1 — tokens
  const tWrap = document.getElementById("attnTokens");
  tWrap.innerHTML = "";
  s.tokens.forEach((t, i) => {
    const el = document.createElement("div");
    el.className = "token";
    el.style.animationDelay = `${i * 25}ms`;
    el.innerHTML = `<span class="text">${escapeHtml(visibleToken(t.text))}</span><span class="id">#${t.id}</span>`;
    tWrap.appendChild(el);
  });

  // Step 2 — Q/K/V token picker
  const picker = document.getElementById("attnTokenPicker");
  picker.innerHTML = "";
  s.tokens.forEach((t, i) => {
    const el = document.createElement("div");
    el.className = "token" + (i === ATTN_SELECTED_TOKEN ? " selected" : "");
    el.innerHTML = `<span class="text">${escapeHtml(visibleToken(t.text))}</span><span class="id">pos ${i}</span>`;
    el.addEventListener("click", () => {
      ATTN_SELECTED_TOKEN = i;
      renderQKV();
    });
    picker.appendChild(el);
  });
  renderQKV();

  // Step 3 — scores (pre-mask, head-specific)
  const scores = s.scores_pre[head];
  Plotly.react("attnScores", [{
    type: "heatmap",
    z: scores, x: labels, y: labels,
    colorscale: SIGNED_COLORSCALE,
    zmin: -Math.max(...scores.flat().map(Math.abs)),
    zmax: Math.max(...scores.flat().map(Math.abs)),
  }], { ...PLOT_LAYOUT, xaxis: { tickangle: -45 } }, PLOT_CONFIG);

  // Step 4 — softmaxed attention (already causal-masked by the model)
  const attn = s.attention[head];
  Plotly.react("attnWeights", [{
    type: "heatmap", z: attn, x: labels, y: labels,
    colorscale: ATTN_COLORSCALE, zmin: 0, zmax: 1,
  }], { ...PLOT_LAYOUT, xaxis: { tickangle: -45 } }, PLOT_CONFIG);

  // Step 5 — output = attention · V (compute client-side using the V we got)
  const T = s.tokens.length;
  const d_head = s.d_head;
  const V = s.v[head]; // T x d_head
  const output = new Array(T).fill(0).map(() => new Array(d_head).fill(0));
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const w = attn[i][j];
      for (let d = 0; d < d_head; d++) {
        output[i][d] += w * V[j][d];
      }
    }
  }
  // Show first 24 dimensions of each output row as heatmap
  const previewDims = Math.min(24, d_head);
  const out = output.map((row) => row.slice(0, previewDims));
  Plotly.react("attnOutput", [{
    type: "heatmap",
    z: out, y: labels,
    x: Array.from({ length: previewDims }, (_, i) => `d${i}`),
    colorscale: SIGNED_COLORSCALE,
    zmin: -Math.max(...out.flat().map(Math.abs)),
    zmax: Math.max(...out.flat().map(Math.abs)),
  }], { ...PLOT_LAYOUT, xaxis: { tickangle: 0 } }, PLOT_CONFIG);
}

function renderQKV() {
  const head = +document.getElementById("attnHead").value;
  const i = ATTN_SELECTED_TOKEN;
  const s = ATTN_STATE;
  const q = s.q[head][i], k = s.k[head][i], v = s.v[head][i];
  document.getElementById("qkvQ").innerHTML = renderVec(q);
  document.getElementById("qkvK").innerHTML = renderVec(k);
  document.getElementById("qkvV").innerHTML = renderVec(v);
  // re-highlight selection
  document.querySelectorAll("#attnTokenPicker .token").forEach((el, idx) => {
    el.classList.toggle("selected", idx === i);
  });
}
function renderVec(v) {
  return v.slice(0, 16).map((x) => `<span class="cell">${x.toFixed(2)}</span>`).join("") +
    (v.length > 16 ? `<span class="cell">…</span>` : "");
}
function visibleToken(t) { return t.replace(/ /g, "·").replace(/\n/g, "↵") || "∅"; }

// Kick off attention with the default sentence after model is ready
function whenModelReady(fn) {
  if (MODEL) fn();
  else setTimeout(() => whenModelReady(fn), 200);
}
whenModelReady(runAttention);

// ---------- Decoding ----------
document.getElementById("genTemp").addEventListener("input", (e) => {
  document.getElementById("genTempVal").textContent = (+e.target.value).toFixed(2);
});
document.getElementById("genTopP").addEventListener("input", (e) => {
  document.getElementById("genTopPVal").textContent = (+e.target.value).toFixed(2);
});
document.getElementById("genGo").addEventListener("click", runGenerate);

async function runGenerate() {
  const prompt = document.getElementById("genPrompt").value;
  const temperature = +document.getElementById("genTemp").value;
  const top_k = +document.getElementById("genTopK").value;
  const top_p = +document.getElementById("genTopP").value;
  const seed = +document.getElementById("genSeed").value;
  try {
    const data = await api("/api/generate", { prompt, temperature, top_k, top_p, seed, n_show: 20 });
    const top = data.top;
    const labels = top.map((t) => visibleToken(t.text));
    Plotly.react("genDist", [{
      type: "bar",
      x: labels, y: top.map((t) => t.prob),
      marker: { color: top.map((t) => t.prob), colorscale: "Blues" },
      text: top.map((t) => `p=${t.prob.toFixed(3)}`),
      textposition: "outside",
    }], {
      ...PLOT_LAYOUT,
      xaxis: { tickangle: -40 },
      yaxis: { title: "P(next token)", gridcolor: "#2a3142" },
    }, PLOT_CONFIG);

    const picks = data.picks;
    const pg = document.getElementById("genPicks");
    pg.innerHTML = `
      <div class="pick">
        <h4>Greedy</h4>
        <span class="picked">${escapeHtml(visibleToken(picks.greedy_text))}</span>
        <div class="meta">arg max — always deterministic</div>
      </div>
      <div class="pick">
        <h4>Top-k (k=${top_k})</h4>
        <span class="picked">${escapeHtml(visibleToken(picks.top_k_text))}</span>
        <div class="meta">sampled from ${picks.top_k_kept} tokens</div>
      </div>
      <div class="pick">
        <h4>Top-p (p=${top_p.toFixed(2)})</h4>
        <span class="picked">${escapeHtml(visibleToken(picks.top_p_text))}</span>
        <div class="meta">nucleus kept ${picks.top_p_kept} tokens</div>
      </div>
    `;
  } catch (e) { alert(e.message); }
}
whenModelReady(runGenerate);

document.getElementById("seqGo").addEventListener("click", async () => {
  const prompt = document.getElementById("genPrompt").value;
  const strategy = document.getElementById("seqStrategy").value;
  const max_new_tokens = +document.getElementById("seqMax").value;
  const temperature = +document.getElementById("genTemp").value;
  const top_k = +document.getElementById("genTopK").value;
  const top_p = +document.getElementById("genTopP").value;
  const seed = +document.getElementById("genSeed").value;
  const out = document.getElementById("seqOutput");
  out.innerHTML = `<span style="color:#9da7b3">generating…</span>`;
  try {
    const data = await api("/api/generate-sequence", {
      prompt, strategy, max_new_tokens, temperature, top_k, top_p, seed,
    });
    out.innerHTML = "";
    const promptSpan = document.createElement("span");
    promptSpan.style.color = "#9da7b3";
    promptSpan.style.marginRight = "8px";
    promptSpan.textContent = prompt;
    out.appendChild(promptSpan);
    data.steps.forEach((s, i) => {
      const t = document.createElement("span");
      t.className = "seq-token";
      t.style.animationDelay = `${i * 50}ms`;
      t.innerHTML = `<span class="text">${escapeHtml(visibleToken(s.text))}</span><span class="prob">${(s.prob * 100).toFixed(1)}%</span>`;
      out.appendChild(t);
    });
  } catch (e) { out.innerHTML = `<span style="color:#f7768e">${e.message}</span>`; }
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));
}
