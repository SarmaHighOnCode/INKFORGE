# INKFORGE — API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required for MVP (v1.0).

---

## Endpoints

### `POST /generate`

Submit a handwriting generation job.

**Request Body:**

```json
{
  "text": "Hello, world!",
  "style_id": "neat_cursive",
  "params": {
    "stroke_width_variation": 0.5,
    "character_inconsistency": 0.4,
    "slant_angle": 5.0,
    "baseline_drift": 0.3,
    "ligature_enabled": true,
    "fatigue_enabled": false,
    "ink_bleed": 0.2
  },
  "paper_texture": "lined",
  "ink_color": "black",
  "font_size": "medium"
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "abc123-def456",
  "ws_url": "ws://localhost:8000/ws/abc123-def456",
  "status": "queued"
}
```

---

### `GET /job/{job_id}`

Poll job status.

**Response (200 OK):**

```json
{
  "job_id": "abc123-def456",
  "status": "processing",
  "progress": 0.45,
  "error": null
}
```

**Status values:** `queued` | `processing` | `complete` | `failed`

---

### `POST /export`

Export completed job to PNG, PDF, or SVG at 300 DPI.

**Request Body:**

```json
{
  "job_id": "abc123-def456",
  "format": "png",
  "paper_size": "a4",
  "transparent_background": false
}
```

**Response (200 OK):**

```json
{
  "download_url": "/exports/abc123-def456.png",
  "format": "png",
  "file_size_bytes": 245760
}
```

---

### `GET /styles`

List all available handwriting style presets.

**Response (200 OK):**

```json
[
  {
    "id": "neat_cursive",
    "name": "Neat Cursive",
    "description": "Clean, flowing cursive handwriting with consistent letter connections."
  },
  {
    "id": "casual_print",
    "name": "Casual Print",
    "description": "Relaxed print handwriting — clear, slightly irregular spacing."
  },
  {
    "id": "rushed_notes",
    "name": "Rushed Notes",
    "description": "Quick, compressed handwriting with visible speed artifacts."
  },
  {
    "id": "doctors_scrawl",
    "name": "Doctor's Scrawl",
    "description": "Highly compressed, barely legible — maximum inconsistency."
  },
  {
    "id": "elegant_formal",
    "name": "Elegant Formal",
    "description": "Deliberate, well-spaced handwriting with slight calligraphic flair."
  }
]
```

---

### `GET /health`

Service health check.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": false,
  "cuda_version": null,
  "pytorch_version": "2.2.0"
}
```

---

## WebSocket — Stroke Streaming

### `WS /ws/{job_id}`

Real-time stroke streaming during generation.

**Message format (server → client):**

```json
{
  "type": "stroke",
  "data": {
    "dx": 2.3,
    "dy": -0.5,
    "p1": 1,
    "p2": 0,
    "p3": 0
  }
}
```

**Completion message:**

```json
{
  "type": "complete",
  "total_strokes": 1247
}
```

---

## Humanization Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `stroke_width_variation` | float | 0.0–1.0 | 0.5 | Pressure variance |
| `character_inconsistency` | float | 0.0–1.0 | 0.4 | Per-char noise in z |
| `slant_angle` | float | -30.0–30.0 | 5.0 | Global slant (degrees) |
| `baseline_drift` | float | 0.0–1.0 | 0.3 | Y-axis noise |
| `ligature_enabled` | bool | — | true | Stroke connections |
| `fatigue_enabled` | bool | — | false | Progressive degradation |
| `ink_bleed` | float | 0.0–1.0 | 0.2 | Edge diffusion |

---

## Enums

### Export Format
`png` | `pdf` | `svg`

### Paper Texture
`lined` | `blank` | `graph` | `aged_parchment`

### Ink Color
`black` | `blue` | `dark_blue` | `sepia`

### Paper Size
`a4` | `us_letter`

### Font Size
`small` (10pt) | `medium` (14pt) | `large` (18pt)
