<p align="center">
  <img src="docs/assets/inkforge-banner.png" alt="Inkforge Banner" width="800" />
</p>

<h1 align="center">✍ INKFORGE</h1>
<h3 align="center">Human-Like Handwriting Synthesis Engine</h3>

<p align="center">
  <strong>Stroke-level generative ML model trained on real human handwriting — not a font.</strong>
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

Every existing text-to-handwriting tool is, under the hood, a **font renderer**. Static handwriting fonts produce a single fixed glyph per character — perfectly uniform stroke width, zero baseline drift, no ligatures, no pressure variance. The human eye detects this inauthenticity instantly.

**Inkforge replaces the font rendering pipeline entirely** with a stroke-level generative ML model trained on real human handwriting corpora.

### Target Users

| Persona | Goal | Pain Point |
|---------|------|------------|
| **Portfolio Builder** | Technically impressive ML project with live demo | Existing generators all use fonts — no real ML differentiator |
| **D2C Marketer** | Personalised handwritten notes in shipments at scale | Font-based tools look fake; robot pens cost $5–8/letter |
| **Real Estate Agent** | Handwritten outreach letters for 3–5x response rates | No tool combines realistic generation with mail pipeline |

---

## The Problem

Real human handwriting is defined by its **imperfections**:

- Pressure builds and releases mid-stroke
- Letters lean inconsistently
- The baseline wanders
- Adjacent characters influence each other through natural ligatures
- Writing degrades in consistency over long passages

**No font can replicate this** because fonts are context-free and deterministic by design.

---

## The Solution

Inkforge synthesizes handwriting as **sequences of pen strokes** with learned distributions over pressure, velocity, slant, and inter-character spacing. Every generation is unique. Every line drifts naturally. Every character is subtly different from its previous instance.

> **This is not a filter applied to a font. It is synthesized handwriting — generated stroke-by-stroke by a deep learning model trained on thousands of real human writers.**

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
| **Fatigue Simulation** | Increasing noise in latent space over token position | On / Off | Off |
| **Ink Bleed** | Post-render Gaussian diffusion on stroke edges | 0.0 – 1.0 | 0.2 |

### User Interface

- **Text Input** — Multi-line, up to 2,000 characters, with paste-from-clipboard support
- **Style Presets** — "Neat Cursive", "Casual Print", "Rushed Notes", "Doctor's Scrawl", "Elegant Formal"
- **Paper Textures** — Lined, Blank, Graph, Aged Parchment
- **Ink Colors** — Black, Blue, Dark Blue, Sepia
- **Live Canvas Preview** — Animated stroke-by-stroke playback with speed control
- **Export** — PNG (300 DPI), PDF (A4/US Letter), SVG (vector)

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
| Style | Concat | Latent `z ∈ ℝ¹²⁸` | Style injection per timestep |
| Encoder L1 | LSTM | hidden=512, dropout=0.2 | Sequence context |
| Encoder L2 | LSTM | hidden=512, dropout=0.2 | Higher-order patterns |
| Encoder L3 | LSTM | hidden=512, dropout=0.2 | Long-range dependencies |
| Output | MDN | M=20 Gaussian mixtures | Stroke distribution sampling |
| Pen State | Bernoulli | Sigmoid × 3 | Pen up/down/end |

**MDN output per timestep:** `(π, μx, μy, σx, σy, ρ, e)` for M=20 components. Temperature `τ` controls generation randomness.

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
| **v1.0** | MVP — Portfolio Launch | LSTM+MDN model, 5 presets, 7 params, PNG/PDF/SVG export, Canvas preview | Week 4 |
| **v1.5** | Style Transfer | CNN reference encoder, handwriting upload, multi-language (Latin) | Week 10 |
| **v2.0** | Diffusion Upgrade | Diffusion backbone, conditional inpainting, quality leap | Week 18 |
| **v2.5** | API & Integrations | Public REST API, Zapier/Make, Lob.com direct mail, bulk endpoint | Week 26 |
| **v3.0** | Enterprise | White-label SDK, custom fine-tuning, on-premise, SLA | Week 40 |

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

<p align="center">
  <strong>Built with ❤️ using PyTorch · FastAPI · React</strong><br/>
  <sub>Inkforge — because handwriting should never be a font.</sub>
</p>
