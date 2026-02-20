# Inkforge — GitHub Copilot Instructions

## Project Context
Inkforge is a handwriting synthesis engine using LSTM+MDN (Mixture Density Network) to generate realistic pen strokes. It is NOT a font renderer. Backend is FastAPI + PyTorch + Celery/Redis. Frontend is React 18 + Vite + Tailwind CSS.

## Stroke Format
All handwriting is encoded as (Δx, Δy, p1, p2, p3) tuples where Δx/Δy are relative pen displacements and p1/p2/p3 are mutually exclusive pen states (down/up/end).

## Code Style
- Python 3.10+ with type hints on all functions
- Google-style docstrings
- PEP 8 with 100-char line limit
- React: functional components + hooks only, Zustand for state
- Tailwind CSS for styling (no inline styles)

## Architecture Rules
- ML model code → `backend/app/ml/`
- Pydantic schemas → `backend/app/models/`
- API routes → `backend/app/api/routes/`
- Always use async for FastAPI routes
- Always queue inference via Celery (never synchronous)
- Model constants: LSTM hidden=512, layers=3, MDN M=20 mixtures, style z∈ℝ¹²⁸

## Don'ts
- Don't use absolute stroke coordinates (always use deltas)
- Don't put ML code in `app/models/` (that's for Pydantic)
- Don't use class-based React components
- Don't commit .env, checkpoints, or data files
