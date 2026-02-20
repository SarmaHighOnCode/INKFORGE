# INKFORGE — Architecture Overview

## System Architecture

Inkforge follows a four-tier architecture with a dedicated document layout engine:

```
┌─────────────────────────────────┐
│        CLIENT (Browser)         │
│   React 18 + Vite + Canvas API  │
│   Tailwind CSS + Zustand        │
│   Full Page Preview + Animation │
└──────────┬──────────────────────┘
           │ HTTPS / WebSocket
           ▼
┌─────────────────────────────────┐
│        API LAYER                │
│   FastAPI + Pydantic            │
│   WebSocket (stroke streaming)  │
└──────────┬──────────────────────┘
           │ Celery Task Queue
           ▼
┌─────────────────────────────────┐
│    DOCUMENT LAYOUT ENGINE       │
│   Paragraph segmentation        │
│   Word-wrap + margin planning   │
│   Line density + spacing calc   │
├─────────────────────────────────┤
│    INFERENCE ENGINE             │
│   PyTorch LSTM+MDN              │
│   TorchScript (optimized)       │
│   Persistent writer state z     │
│   Position-aware fatigue model  │
├─────────────────────────────────┤
│    RENDERER                     │
│   CairoSVG + Pillow             │
│   Full page SVG → PNG/PDF      │
└─────────────────────────────────┘
           │
       Redis (Task Broker + Result Store)
```

## Model Architecture

See [PRD Section 4.2](PRD.md#422-core-model-lstmmdn) for full specification.

### LSTM + Mixture Density Network

```
Input (char one-hot)  →  Embedding (d=256)
                              ↓
Style vector z ∈ ℝ¹²⁸  →  Concatenate
Page position encoding  →  Concatenate
                              ↓
                        LSTM Layer 1 (hidden=512, dropout=0.2)
                              ↓
                        LSTM Layer 2 (hidden=512, dropout=0.2)
                              ↓
                        LSTM Layer 3 (hidden=512, dropout=0.2)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            MDN Head (M=20)      Pen State Head
         (π, μx, μy, σx, σy, ρ)   (p1, p2, p3)
```

Key difference from single-line generators: the **style vector z** and **page position encoding** persist across lines, giving the model document-level context. The LSTM hidden state carries accumulated fatigue and drift information.

### Stroke Tuple Format

```
(Δx, Δy, p₁, p₂, p₃)

Δx, Δy = relative pen displacements from previous position
p₁     = pen-down (actively drawing)
p₂     = pen-up   (moving without drawing)
p₃     = end-of-sequence sentinel
```

## Data Flow — 4-Step Generation Pipeline

### Overview

1. User submits full document text + style + params via React UI
2. FastAPI validates and queues Celery task
3. Celery worker executes the **4-step pipeline** below
4. Strokes streamed back via WebSocket (full page)
5. Canvas API renders full-page preview
6. Export endpoint re-renders complete document at 300 DPI

### Step 1 — Smart Text Chunking

The input text is split into **individual words or short phrases**. Each word becomes a separate inference unit — but they are **not independent** (see Step 2).

### Step 2 — LSTM Hidden State Passing

When the model generates strokes for Word 1, the **final LSTM hidden state `h_t`** is captured and passed as the **initial state for Word 2**. This chain runs across the entire document, ensuring every word inherits the accumulated writer personality, slant habits, pressure tendencies, and fatigue.

```
"Thank" → LSTM → strokes + h₁
"you"   → LSTM(init=h₁) → strokes + h₂
"for"   → LSTM(init=h₂) → strokes + h₃
  ... entire document
```

### Step 3 — 2D Typewriter Layout Algorithm

A **classical Python script** (outside the ML model) places each generated word on the page like a typewriter:

- Place word at cursor `(x, y)`
- Advance `x` by `word_width + random_space(base=10px, noise=±3px)`
- If `x > right_margin` → line break: reset `x`, shift `y` down with random noise
- If paragraph break → extra `y` shift + paragraph indent with natural variation

### Step 4 — Global Sine-Wave Baseline Drift

A slow sinusoidal function is applied to all y-positions across the page:

```
y_offset(line_n) = A × sin(2π × line_n / period + phase)
```

This makes lines gently curve up and down (A=1–3px, period=8–12 lines), layered **on top of** the per-word jitter from the LSTM.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `backend/app/ml/` | PyTorch model definitions |
| `backend/app/api/routes/` | FastAPI endpoint handlers |
| `backend/app/services/` | Business logic (inference, rendering, document layout) |
| `backend/app/models/` | Pydantic request/response schemas |
| `frontend/src/components/` | React UI components |
| `configs/` | Training and inference YAML configs |
| `scripts/` | Data download and preprocessing |
