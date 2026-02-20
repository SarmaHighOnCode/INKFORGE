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
  "text": "Dear Ms. Johnson,\n\nThank you for taking the time to meet with us last Thursday. We truly appreciate your insights on the neighborhood and your commitment to finding the right home for your family.\n\nAs discussed, I've prepared a shortlist of three properties that match your criteria. Each one offers the open floor plan and backyard space you mentioned, and they're all within walking distance of Lincoln Elementary.\n\nI'd love to schedule viewings at your earliest convenience. Please don't hesitate to reach out if you have any questions in the meantime.\n\nWarm regards,\nJames",
  "style_id": "neat_cursive",
  "params": {
    "stroke_width_variation": 0.5,
    "character_inconsistency": 0.4,
    "slant_angle": 5.0,
    "baseline_drift": 0.3,
    "ligature_enabled": true,
    "fatigue_simulation": 0.3,
    "ink_bleed": 0.2
  },
  "layout": {
    "page_size": "a4",
    "margin_left_cm": 2.0,
    "margin_right_cm": 1.5,
    "margin_top_cm": 2.5,
    "margin_bottom_cm": 2.0,
    "line_spacing_mm": 8.0,
    "paragraph_indent_cm": 1.5,
    "paragraph_spacing_multiplier": 1.2
  },
  "paper_texture": "blank",
  "ink_color": "blue",
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
| `fatigue_simulation` | float | 0.0–1.0 | 0.3 | Progressive degradation over long passages |
| `ink_bleed` | float | 0.0–1.0 | 0.2 | Edge diffusion |

---

## Document Layout Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page_size` | string | `"a4"` | Page dimensions (`a4` or `us_letter`) |
| `margin_left_cm` | float | 2.0 | Left margin in cm (with natural drift) |
| `margin_right_cm` | float | 1.5 | Right margin in cm |
| `margin_top_cm` | float | 2.5 | Top margin in cm |
| `margin_bottom_cm` | float | 2.0 | Bottom margin in cm |
| `line_spacing_mm` | float | 8.0 | Base inter-line spacing in mm (varies ±0.5mm naturally) |
| `paragraph_indent_cm` | float | 1.5 | First-line indent per paragraph (with natural variation) |
| `paragraph_spacing_multiplier` | float | 1.2 | Vertical spacing between paragraphs as multiplier of line height |

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
