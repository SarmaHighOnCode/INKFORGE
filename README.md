<p align="center">
  <img src="docs/assets/inkforge-banner.png" alt="Inkforge Banner" width="800" />
</p>

<h1 align="center">✍ INKFORGE</h1>
<h3 align="center">Long-Form Human Handwriting Synthesis Engine</h3>

<p align="center">
  <strong>Generate full pages of realistic handwriting — paragraphs, essays, letters — with natural fatigue, drift, and writer personality. Not a font. Not a single-line demo. A complete document-level handwriting engine.</strong>
</p>

<p align="center">
  <a href="https://github.com/SarmaHighOnCode/INKFORGE/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/SarmaHighOnCode/INKFORGE/ci.yml?style=flat-square" alt="Build Status" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</p>

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Features (MVP)](#features-mvp)
- [Technical Architecture](#technical-architecture)
- [Data Flow](#data-flow)
- [Getting Started](#getting-started)
- [Docker Compose (All-in-One)](#docker-compose-all-in-one)
- [Roadmap](#roadmap)
- [KPIs & Success Metrics](#kpis--success-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Executive Summary

Every existing text-to-handwriting tool falls into one of two traps:

1. **Font renderers** — Static handwriting fonts that produce identical glyphs every time. The human eye detects this instantly.
2. **Single-line demo generators** (e.g., Calligrapher AI) — RNN-based tools that generate one short line at a time with basic style controls. They can produce a convincing sentence, but ask them for a full page? They have no concept of paragraphs, margins, page layout, or the way human handwriting evolves over long passages.

**Inkforge is different.** It generates **entire documents** — full paragraphs, multi-page letters, long-form essays — where the handwriting looks like it was written by a real human sitting at a desk for 20 minutes, not generated one line at a time and stitched together.

### How Inkforge Differs from Calligrapher AI & Others

| Feature | Calligrapher AI / Font Tools | **Inkforge** |
|---------|-----------------------------|--------------|
| Scope | Single line or short snippet | **Full documents — paragraphs, pages, essays** |
| Line Wrapping | None — user manually splits lines | **Automatic word-wrap with natural margin awareness** |
| Writing Fatigue | None | **Progressive degradation over long passages** — stroke quality, size, spacing all evolve |
| Paragraph Structure | None | **Indentation, paragraph spacing, section breaks** |
| Inter-line Consistency | Each line generated independently | **Lines are coherent within a page** — consistent writer personality with natural drift |
| Character Memory | Stateless per generation | **Writer-consistent evolution** — the same character looks subtly different each time, but consistently "from the same hand" |
| Page Layout | Not applicable | **Full page composition** — margins, headers, line spacing, multi-page support |
| Output Length | ~1 line (typically <100 chars) | **Up to 2,000+ characters** — full page A4/Letter output |
| Export | SVG only | **PNG (300 DPI), PDF (A4/US Letter), SVG** |

### Target Users

| Persona | Goal | Pain Point |
|---------|------|------------|
| **Portfolio Builder** | Technically impressive ML project with live demo | Existing generators produce short demos — no long-form document generation |
| **D2C Marketer** | Personalised handwritten notes in shipments at scale | Font-based tools look fake; single-line generators can't produce full letters |
| **Real Estate Agent** | Handwritten outreach letters for 3–5x response rates | No tool generates a convincing full-page handwritten letter |
| **Student / Creator** | Handwritten essays, assignments, or journal pages | Need realistic multi-paragraph output, not one line at a time |

---

## The Problem

Real human handwriting over a **full page** is defined by its **imperfections at every scale**:

### Character Level
- Pressure builds and releases mid-stroke
- Letters lean inconsistently
- The same letter is written slightly differently every time — but consistently "from the same hand"

### Line Level
- The baseline wanders across a line
- Adjacent characters influence each other through natural ligatures
- Word spacing varies naturally — tighter in fast sections, looser in deliberate ones

### Document Level (what no other tool handles)
- Writing quality **degrades over long passages** — fatigue is real
- Letter size subtly **grows or shrinks** over paragraphs
- Margins aren't perfectly straight — the left edge drifts
- Line spacing isn't uniform — it loosens as the writer reaches the bottom of a page
- Paragraph indentation varies between paragraphs
- The overall slant may shift across the page as the writer's hand position changes

**No font can replicate this.** And no single-line generator even attempts it.

---

## The Solution

Inkforge synthesizes handwriting as **sequences of pen strokes** with learned distributions over pressure, velocity, slant, and inter-character spacing — but unlike short-snippet generators, it operates at the **document level**.

When you feed Inkforge a 500-word essay:
- It plans the **page layout** — margins, line count, paragraph breaks
- It generates each line within the context of the **full document** — the model knows where it is on the page
- It simulates **writing fatigue** — the 30th line isn't as crisp as the 1st
- It maintains **writer consistency** — every character comes from the same "hand", with natural per-instance variation
- Every generation is **unique** — regenerating the same text produces a completely different manuscript

> **This is not a filter applied to a font. This is not a single-line demo. It is a full document synthesis engine — generating pages of handwriting stroke-by-stroke, with the realism of a human writer sitting at a desk.**

---

## Features (MVP)

### ML Humanization Parameters

Each parameter is implemented at the **model level**, not as post-processing. They map to dimensions of the learned stroke distribution and are exposed via UI sliders.

| Parameter | Technical Implementation | UI Range | Default |
|-----------|------------------------|----------|---------|
| **Stroke Width Variation** | Pressure derived from pen velocity — fast strokes → thin; slow → wide | 0.0 – 1.0 | 0.5 |
| **Character Inconsistency** | Per-character noise injected into style latent vector `z` | 0.0 – 1.0 | 0.4 |
| **Slant Angle** | Global slant bias + per-word variance from learned distribution | -30° to +30° | 5° |
| **Baseline Drift** | Slow-varying sinusoidal noise on y-axis across a line | 0.0 – 1.0 | 0.3 |
| **Ligature Formation** | Contextual stroke connections between adjacent characters | On / Off | On |
| **Fatigue Simulation** | Progressive degradation over long passages — stroke precision decreases, letter size drifts, spacing loosens | 0.0 – 1.0 | 0.3 |
| **Ink Bleed** | Post-render Gaussian diffusion on stroke edges | 0.0 – 1.0 | 0.2 |

### Document-Level Layout Features

These features are what separate Inkforge from single-line generators. They enable full-page, multi-paragraph output.

| Feature | Description | Default |
|---------|-------------|---------|
| **Auto Line Wrapping** | Text automatically wraps at page margins with natural word-boundary detection | On |
| **Paragraph Indentation** | First line of each paragraph indented with natural variation | 1.5 cm ± natural drift |
| **Paragraph Spacing** | Variable vertical spacing between paragraphs | 1.2× line height |
| **Margin Awareness** | Left/right margins with natural drift — not ruler-straight | 2 cm ± subtle variation |
| **Inter-line Spacing** | Line spacing varies subtly across the page — loosens toward bottom | ~8mm ± 0.5mm drift |
| **Page Composition** | Full A4/Letter page layout with configurable margins and line density | A4, 25–30 lines/page |
| **Writer Hand Position Shift** | Slant and baseline shift as the writer's hand moves down the page | Subtle, progressive |

### User Interface

- **Text Input** — Multi-line, up to 2,000+ characters, with paste-from-clipboard support. Supports full paragraphs, essays, and letters
- **Style Presets** — "Neat Cursive", "Casual Print", "Rushed Notes", "Doctor's Scrawl", "Elegant Formal"
- **Paper Textures** — Lined, Blank, Graph, Aged Parchment
- **Ink Colors** — Black, Blue, Dark Blue, Sepia
- **Live Canvas Preview** — Full page animated stroke-by-stroke playback with speed control
- **Page Preview** — WYSIWYG preview showing exactly how the full document will look on paper
- **Export** — PNG (300 DPI), PDF (A4/US Letter), SVG (vector) — full page output, not just a single line

---

## Technical Architecture

### System Overview

```
CLIENT (React + Canvas)
    │
    ▼ HTTPS / WebSocket
API LAYER (FastAPI)
    │
    ▼ Async Task
TASK QUEUE (Celery + Redis)
    │
    ▼ Inference
ML ENGINE (PyTorch LSTM+MDN)
    │
    ▼ Render
RENDERER (CairoSVG + Pillow)
    │
    ▼ Export
STORAGE (PNG / PDF / SVG)
```

### Core Model — LSTM + Mixture Density Network

Based on [Graves (2013)](https://arxiv.org/abs/1308.0850) — *Generating Sequences with Recurrent Neural Networks*.

**Stroke Representation** — Each handwriting sample is encoded as a sequence of 5-tuples:

```
(Δx, Δy, p₁, p₂, p₃)

Δx, Δy = relative pen displacements
p₁     = pen-down (actively drawing)
p₂     = pen-up   (moving without drawing)
p₃     = end-of-sequence sentinel
```

**Model Layers:**

| Layer | Type | Configuration | Purpose |
|-------|------|---------------|---------|
| Input | Embedding | Char one-hot → d=256 | Character encoding |
| Style | Concat | Latent `z ∈ ℝ¹²⁸` | Writer personality vector |
| Position | Encoding | Page position (line #, char position) | Document-level context |
| Encoder L1 | LSTM | hidden=512, dropout=0.2 | Sequence context |
| Encoder L2 | LSTM | hidden=512, dropout=0.2 | Higher-order patterns |
| Encoder L3 | LSTM | hidden=512, dropout=0.2 | Long-range dependencies (cross-line coherence) |
| Output | MDN | M=20 Gaussian mixtures | Stroke distribution sampling |
| Pen State | Bernoulli | Sigmoid × 3 | Pen up/down/end |

**MDN output per timestep:** `(π, μx, μy, σx, σy, ρ, e)` for M=20 components. Temperature `τ` controls generation randomness.

**Document-Level Generation Pipeline — How It Actually Works:**

Unlike single-line generators (Calligrapher AI, etc.) that generate each line in isolation and discard all state, Inkforge uses a **4-step pipeline** that maintains writer consistency across the entire document:

#### Step 1 — Smart Text Chunking

The full input text (up to 2,000+ characters) is broken into **individual words or short phrases**. Each chunk becomes a separate inference call to the LSTM — but critically, these calls are **not independent**.

```
"Thank you for meeting with us last Thursday."
    │
    ▼ Tokenizer
["Thank", "you", "for", "meeting", "with", "us", "last", "Thursday."]
```

#### Step 2 — LSTM State Passing (The Secret Sauce)

This is what makes Inkforge fundamentally different from tools that generate text line-by-line. When the model generates strokes for "Word 1", the **final LSTM hidden state `h_t`** is captured and used as the **initial hidden state for "Word 2"**.

```
Word 1: "Thank"
    LSTM processes → generates strokes → final state h₁
                                              │
Word 2: "you"                                 │
    LSTM starts with h₁ → generates strokes → final state h₂
                                              │
Word 3: "for"                                 │
    LSTM starts with h₂ → generates strokes → final state h₃
                                              │
    ... and so on for the entire document
```

**Why this matters:** The hidden state carries all the accumulated "writer personality" — slant tendencies, pressure habits, letter-formation quirks, and fatigue. Every word inherits the full writing history, so word 50 naturally looks like it was written by the same hand that wrote word 1 — just a bit more tired.

#### Step 3 — 2D Typewriter Layout Algorithm

A **classical Python layout engine** (outside the ML model) acts like a typewriter to place each generated word on the page:

```
For each generated word chunk:
    1. Measure the rendered stroke width of the word
    2. Place it at current cursor position (x, y)
    3. Advance x by: word_width + random_space(base=10px, noise=±3px)
    4. If x > right_margin:
         → Line break: reset x to left_margin + slight_random_offset
         → Shift y down by: line_height + random_noise(±0.5mm)
         → Apply subtle baseline drift to new line
    5. If paragraph break detected:
         → Extra y shift (1.2× line height)
         → Apply paragraph indent to x (1.5cm ± natural variation)
```

This keeps all layout logic **deterministic and debuggable** — no ML model is wasting capacity learning where to put spaces and line breaks.

#### Step 4 — Global Baseline Variance (Sine-Wave Drift)

A slow-moving **sinusoidal function** is applied to the y-axis across the entire page, making lines gently curve up and down rather than sitting on perfectly ruled baselines:

```
y_offset(line_n) = A × sin(2π × line_n / period + phase)

Where:
    A      = amplitude (1–3px) — subtle enough to look natural
    period = 8–12 lines — one full wave across ~half a page
    phase  = random per generation — so no two pages curve the same way
```

This is applied **on top of** the per-line baseline drift from the LSTM, creating two layers of natural variation: the model's own stroke-level jitter, plus a global page-level undulation that mimics how a human's hand position shifts as they write down a page.

**Training Data:** [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) — 13,049 texts, 221 writers. Writer-level train/val/test split (80/10/10).

### Backend Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| API | FastAPI 0.110+ | Async-native, auto OpenAPI docs, WebSocket |
| Model Serving | PyTorch 2.x + TorchScript | Optimized CPU/GPU inference |
| Task Queue | Celery + Redis | Async jobs prevent timeout |
| Rendering | CairoSVG + Pillow | SVG → PNG/PDF at 300 DPI |
| Container | Docker + Compose | Reproducible environments |

### Frontend Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Framework | React 18 + Vite 5 | Fast HMR, component-based |
| Canvas | HTML5 Canvas API | Stroke-by-stroke animation |
| State | Zustand | Lightweight, no boilerplate |
| HTTP/WS | Axios + React Query | Caching, loading states |
| Styling | Tailwind CSS 3 | Utility-first, rapid prototyping |

---

## Data Flow

### End-to-End Request Lifecycle

| Step | Component | Action | Output |
|------|-----------|--------|--------|
| 1 | React UI | User types text, adjusts sliders, selects style | `{ text, style_id, params }` |
| 2 | Axios Client | `POST /generate` + WebSocket opened | `202 Accepted` + WS URL |
| 3 | Celery + Redis | Request queued as async task | Job ID |
| 4 | FastAPI Worker | Text tokenized, params normalized | Token array + param vector |
| 5 | Style Encoder | `style_id` → precomputed `z ∈ ℝ¹²⁸` | Latent vector |
| 6 | LSTM + MDN | Autoregressive stroke sampling at `τ=f(inconsistency)` | Stroke sequence |
| 7 | WebSocket | Strokes streamed to frontend in real-time | Chunked stroke data |
| 8 | Canvas API | Strokes drawn with variable width on paper texture | Animated preview |
| 9 | FastAPI | `POST /export` → re-render at 300 DPI | Download URL |

### API Endpoints (MVP)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Submit generation job → returns `job_id` + WebSocket URL |
| `GET` | `/job/{job_id}` | Poll job status: `queued \| processing \| complete \| failed` |
| `POST` | `/export` | Re-render at print resolution → returns download URL |
| `GET` | `/styles` | List all available style presets |
| `GET` | `/health` | Service health + model load status + GPU availability |

---

## Getting Started

### Prerequisites

| Dependency | Min Version | Purpose |
|-----------|-------------|---------|
| Python | 3.10+ | Backend runtime |
| Node.js | 18+ | Frontend build (Vite + React) |
| Redis | 7.x | Celery task broker |
| CUDA (optional) | 11.8+ | GPU acceleration for training |
| Docker + Compose | Latest | Optional all-in-one setup |

### Setup

```bash
# 1. Clone
git clone https://github.com/SarmaHighOnCode/INKFORGE.git
cd INKFORGE

# 2. Backend
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download or train the model
python scripts/download_iam.py --output data/iam/
python train.py --config configs/lstm_mdn_base.yaml
# OR download pretrained:
python scripts/download_checkpoint.py --model lstm_mdn_v1

# 4. Start FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Frontend (new terminal)
cd frontend
npm install
cp .env.example .env.local      # Set VITE_API_URL=http://localhost:8000
npm run dev

# 6. Redis (new terminal)
docker run -d -p 6379:6379 redis:alpine

# 7. Celery worker (new terminal)
cd backend
celery -A app.worker worker --loglevel=info

# 8. Open browser
# http://localhost:5173
```

---

## Docker Compose (All-in-One)

```bash
docker compose up --build
```

Starts all services (API, Celery worker, Redis, frontend) in one command. Recommended for first-time setup and demo environments.

---

## Roadmap

| Version | Milestone | Key Deliverables | Target |
|---------|-----------|-----------------|--------|
| **v1.0** | MVP — Long-Form Generation | LSTM+MDN model, full-page output, document layout engine, 5 presets, 7 params, fatigue simulation, PNG/PDF/SVG export, Canvas preview | Week 4 |
| **v1.5** | Style Transfer + Upload | CNN reference encoder, handwriting sample upload for custom style cloning, multi-language (Latin) | Week 10 |
| **v2.0** | Diffusion Upgrade | Diffusion backbone, conditional inpainting, quality leap for ultra-long documents | Week 18 |
| **v2.5** | API & Integrations | Public REST API, bulk generation (100+ letters), Zapier/Make, Lob.com direct mail pipeline | Week 26 |
| **v3.0** | Enterprise | White-label SDK, custom fine-tuning on client handwriting, on-premise, SLA | Week 40 |

---

## KPIs & Success Metrics

| KPI | Target | Measurement |
|-----|--------|-------------|
| Generation Latency | < 3s / 100 chars (CPU) | API p95 response time |
| Output Realism | > 75% human-pass rate | Blind A/B test (n=20) |
| Export Quality | 300 DPI, no artefacts | Manual QA on 50 samples |
| Demo Uptime | 99%+ | UptimeRobot |
| GitHub Stars | 50+ in 30 days | Repo analytics |
| CI Pass Rate | 100% on main | GitHub Actions |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## References

- Graves, A. (2013). [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850). arXiv:1308.0850
- [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [CairoSVG Documentation](https://cairosvg.org)

---

## How We Differ from Existing Tools

> **Calligrapher AI** and similar tools are impressive single-line demos. They generate a short sentence with style controls — and that's where they stop.
>
> **Inkforge generates documents.** Feed it an entire essay and get back a realistic handwritten manuscript — with natural paragraph breaks, margin awareness, writing fatigue, and the kind of page-level coherence that only comes from treating the document as a whole, not as a collection of independent lines.

---

<p align="center">
  <strong>Built with ❤️ using PyTorch · FastAPI · React</strong><br/>
  <sub>Inkforge — because handwriting should never be a font, and a real letter is more than one line.</sub>
</p>
