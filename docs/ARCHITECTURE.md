# INKFORGE — Architecture Overview

## System Architecture

Inkforge follows a three-tier architecture:

```
┌─────────────────────────────────┐
│        CLIENT (Browser)         │
│   React 18 + Vite + Canvas API  │
│   Tailwind CSS + Zustand        │
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
│    INFERENCE ENGINE             │
│   PyTorch LSTM+MDN              │
│   TorchScript (optimized)       │
├─────────────────────────────────┤
│    RENDERER                     │
│   CairoSVG + Pillow             │
│   SVG → PNG/PDF at 300 DPI     │
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

### Stroke Tuple Format

```
(Δx, Δy, p₁, p₂, p₃)

Δx, Δy = relative pen displacements from previous position
p₁     = pen-down (actively drawing)
p₂     = pen-up   (moving without drawing)
p₃     = end-of-sequence sentinel
```

## Data Flow

1. User submits text + style + params via React UI
2. FastAPI validates and queues Celery task
3. Celery worker runs LSTM+MDN inference
4. Strokes streamed back via WebSocket
5. Canvas API renders with variable stroke width
6. Export endpoint re-renders at 300 DPI

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `backend/app/ml/` | PyTorch model definitions |
| `backend/app/api/routes/` | FastAPI endpoint handlers |
| `backend/app/services/` | Business logic (inference, rendering) |
| `backend/app/models/` | Pydantic request/response schemas |
| `frontend/src/components/` | React UI components |
| `configs/` | Training and inference YAML configs |
| `scripts/` | Data download and preprocessing |
