# INKFORGE — AI Development Guidelines

> **These guidelines are for AI coding assistants (GitHub Copilot, Cursor, Cline, Claude, etc.)
> working on the Inkforge codebase.** Follow these conventions strictly.

---

## 1. Project Overview

**Inkforge** is a human-like handwriting synthesis engine powered by a stroke-level generative ML model (LSTM + Mixture Density Network). It is **not** a font renderer. The system generates handwriting as sequences of pen strokes with learned distributions over pressure, velocity, slant, and spacing.

### Architecture (3-Tier)

```
React Frontend → FastAPI Backend → PyTorch Inference Engine
                     ↓
              Celery + Redis (async task queue)
                     ↓
              CairoSVG + Pillow (rendering)
```

### Key Reference

- **Paper:** Graves (2013) — "Generating Sequences with Recurrent Neural Networks" (arXiv:1308.0850)
- **Dataset:** IAM On-Line Handwriting Database (13,049 texts, 221 writers)

---

## 2. Stroke Representation (CRITICAL)

All handwriting is represented as sequences of **5-tuples**:

```
(Δx, Δy, p₁, p₂, p₃)

Δx, Δy = relative pen displacements from previous position
p₁     = pen-down (actively drawing)
p₂     = pen-up   (moving without drawing)
p₃     = end-of-sequence sentinel
```

**Rules:**
- Exactly one of `p₁, p₂, p₃` is 1 at any timestep; the others are 0
- `Δx, Δy` are relative (delta) coordinates, NOT absolute
- When converting to absolute for rendering, accumulate deltas
- Stroke sequences are variable-length; pad/truncate to `max_seq_len=700` for training

---

## 3. Model Architecture Constants

Do NOT change these values without explicit approval — they are baked into the PRD:

| Parameter | Value | Location |
|-----------|-------|----------|
| Character embedding dim | d=256 | `model.py` |
| Style latent dim | z ∈ ℝ¹²⁸ | `model.py`, `style_encoder.py` |
| LSTM hidden dim | 512 | `model.py` |
| LSTM layers | 3 | `model.py` |
| Dropout | 0.2 | `model.py` |
| MDN mixtures (M) | 20 | `model.py` |
| MDN params per mixture | 6 (π, μx, μy, σx, σy, ρ) | `model.py` |
| Pen state outputs | 3 (p₁, p₂, p₃) | `model.py` |

---

## 4. Humanization Parameters

These 7 parameters are exposed to users via UI sliders. They are NOT post-processing — they operate at the model/latent level:

| Parameter | Default | Range | Implementation |
|-----------|---------|-------|----------------|
| Stroke Width Variation | 0.5 | 0.0–1.0 | Derived from pen velocity |
| Character Inconsistency | 0.4 | 0.0–1.0 | Noise in style vector z |
| Slant Angle | 5° | -30° to +30° | Global bias + per-word variance |
| Baseline Drift | 0.3 | 0.0–1.0 | Sinusoidal y-axis noise |
| Ligature Formation | Enabled | On/Off | Contextual stroke connections |
| Fatigue Simulation | Disabled | On/Off | Increasing latent noise over position |
| Ink Bleed | 0.2 | 0.0–1.0 | Post-render Gaussian diffusion |

---

## 5. Python Code Style (Backend + ML)

### General
- **Python 3.10+** — use modern type hints (`list[str]`, `dict[str, int]`, `X | None`)
- **PEP 8** — enforced via `ruff`
- **Line length:** 100 characters max
- **Imports:** sorted with `isort` (ruff handles this)

### Type Hints
```python
# ✅ Good — all args and returns typed
def generate(self, text: str, style_z: torch.Tensor, temperature: float = 0.4) -> list[tuple]:
    ...

# ❌ Bad — missing types
def generate(self, text, style_z, temperature=0.4):
    ...
```

### Docstrings (Google Style)
```python
def compute_mdn_loss(
    mdn_params: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MDN negative log-likelihood loss.

    Args:
        mdn_params: Predicted mixture parameters [batch, seq, M*6].
        target: Ground truth strokes [batch, seq, 2].

    Returns:
        Scalar loss tensor.
    """
```

### Pydantic Models
- Use `pydantic.BaseModel` for all API schemas
- Use `Field(...)` with descriptions for all fields
- Use enums for fixed choice sets
- Validate constraints with `ge`, `le`, `min_length`, `max_length`

### FastAPI Patterns
- Use `APIRouter` per domain (generate, export, styles, health)
- All route functions must be `async`
- Use dependency injection for services
- Return proper HTTP status codes (202 for async jobs, 404 for not found)

---

## 6. JavaScript/JSX Code Style (Frontend)

- **React 18** with functional components and hooks only (no class components)
- **Zustand** for state management (no Redux)
- **Tailwind CSS** for styling (utility-first)
- Use `const` by default; `let` only when reassignment is needed
- Destructure props and state
- File naming: `PascalCase.jsx` for components, `camelCase.js` for utils/hooks/stores

### Component Structure
```jsx
// 1. Imports
import { useState, useEffect } from "react";

// 2. Component
function TextInputPanel({ onTextChange, maxLength = 2000 }) {
  const [text, setText] = useState("");
  
  // 3. Handlers
  const handleChange = (e) => {
    // ...
  };

  // 4. Render
  return (
    <div>...</div>
  );
}

// 5. Export
export default TextInputPanel;
```

---

## 7. File Organization Rules

```
backend/
  app/
    api/routes/       → One file per endpoint group
    models/           → Pydantic schemas only (NOT ML models)
    services/         → Business logic (inference, rendering)
    ml/               → PyTorch model definitions and training code
  tests/              → Mirror app/ structure with test_ prefix

frontend/
  src/
    components/       → React components (PascalCase.jsx)
    hooks/            → Custom hooks (useXxx.js)
    stores/           → Zustand stores (xxxStore.js)
    utils/            → Helper functions (camelCase.js)
    assets/           → Static assets (images, icons)
```

**Rules:**
- Never put ML model code in `models/` (that's for Pydantic schemas)
- ML code goes in `app/ml/`
- One React component per file
- Keep components under 200 lines; extract sub-components if longer

---

## 8. API Conventions

### Endpoints (MVP)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/generate` | Submit async generation job |
| GET | `/job/{job_id}` | Poll job status |
| POST | `/export` | Render to PNG/PDF/SVG |
| GET | `/styles` | List style presets |
| GET | `/health` | Service health check |

### Response Format
- Always return JSON
- Use `202 Accepted` for async jobs (not 200)
- Include `job_id` in generation responses
- Error responses must include `detail` field

---

## 9. Git & Commit Conventions

### Branch Naming
- `feat/` — new features
- `fix/` — bug fixes
- `refactor/` — code restructuring
- `docs/` — documentation
- `ml/` — ML model changes

### Commit Messages (Conventional Commits)
```
feat(api): add WebSocket stroke streaming endpoint
fix(ml): correct MDN loss gradient computation
docs: update README with training instructions
refactor(frontend): extract CanvasPreview component
```

---

## 10. Testing Requirements

- **Backend:** pytest with `pytest-asyncio` for async endpoints
- **ML:** Test model instantiation, output shapes, and MDN sampling
- **API:** Use `TestClient` from FastAPI
- All new features must include tests
- Maintain >80% coverage on core modules

---

## 11. Common Pitfalls — AVOID THESE

1. **DO NOT** use absolute coordinates for strokes — always use deltas `(Δx, Δy)`
2. **DO NOT** treat this as a font rendering system — strokes are generated, not looked up
3. **DO NOT** put ML model Python code in `app/models/` — that's for Pydantic schemas
4. **DO NOT** use `any` type in TypeScript/JavaScript — use proper types
5. **DO NOT** commit model checkpoints (`.pt`, `.pth`) — they are gitignored
6. **DO NOT** commit `.env` files — only `.env.example`
7. **DO NOT** hardcode model hyperparameters — use config YAML files
8. **DO NOT** use synchronous inference in API routes — always queue via Celery
9. **DO NOT** mix pen states — exactly one of `(p₁, p₂, p₃)` must be 1 at each timestep
10. **DO NOT** use class-based React components — only functional + hooks

---

## 12. Security & Ethics

- Never generate content that simulates signatures
- Include watermark metadata in all exports
- Sanitize all user text input before processing
- Rate-limit generation endpoints (future: API key auth)
- No PII stored in generation artifacts
