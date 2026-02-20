"""
INKFORGE — Pydantic Request/Response Schemas

Defines all API data models for request validation and response serialization.
"""

from enum import Enum

from pydantic import BaseModel, Field


# ============================================================
# Enums
# ============================================================

class ExportFormat(str, Enum):
    """Supported export formats."""
    PNG = "png"
    PDF = "pdf"
    SVG = "svg"


class PaperTexture(str, Enum):
    """Available paper textures."""
    LINED = "lined"
    BLANK = "blank"
    GRAPH = "graph"
    AGED_PARCHMENT = "aged_parchment"


class InkColor(str, Enum):
    """Available ink colors."""
    BLACK = "black"
    BLUE = "blue"
    DARK_BLUE = "dark_blue"
    SEPIA = "sepia"


class PaperSize(str, Enum):
    """Supported paper sizes for PDF export."""
    A4 = "a4"
    US_LETTER = "us_letter"


class FontSize(str, Enum):
    """Font size equivalents."""
    SMALL = "small"      # ~10pt
    MEDIUM = "medium"    # ~14pt
    LARGE = "large"      # ~18pt


# ============================================================
# Humanization Parameters
# ============================================================

class HumanizationParams(BaseModel):
    """
    ML humanization parameters — each maps to a dimension
    of the learned stroke distribution.
    """
    stroke_width_variation: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Pressure variance — fast strokes thin, slow strokes wide.",
    )
    character_inconsistency: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Per-character noise in style latent vector z.",
    )
    slant_angle: float = Field(
        default=5.0, ge=-30.0, le=30.0,
        description="Global slant bias in degrees, with per-word variance.",
    )
    baseline_drift: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Slow-varying sinusoidal noise on y-axis.",
    )
    ligature_enabled: bool = Field(
        default=True,
        description="Enable contextual stroke connections between adjacent characters.",
    )
    fatigue_enabled: bool = Field(
        default=False,
        description="Increasing noise over token position — simulates writing fatigue.",
    )
    ink_bleed: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Post-render Gaussian diffusion on stroke edges.",
    )


# ============================================================
# Style Preset
# ============================================================

class StylePreset(BaseModel):
    """A preloaded handwriting style preset."""
    id: str
    name: str
    description: str


# ============================================================
# Request Models
# ============================================================

class GenerateRequest(BaseModel):
    """Request body for POST /generate."""
    text: str = Field(
        ..., min_length=1, max_length=2000,
        description="Input text to synthesize (max 2,000 characters).",
    )
    style_id: str = Field(
        default="neat_cursive",
        description="ID of the style preset to use.",
    )
    params: HumanizationParams = Field(
        default_factory=HumanizationParams,
        description="Humanization parameter overrides.",
    )
    paper_texture: PaperTexture = Field(default=PaperTexture.LINED)
    ink_color: InkColor = Field(default=InkColor.BLACK)
    font_size: FontSize = Field(default=FontSize.MEDIUM)


class ExportRequest(BaseModel):
    """Request body for POST /export."""
    job_id: str = Field(..., description="ID of the completed generation job.")
    format: ExportFormat = Field(default=ExportFormat.PNG)
    paper_size: PaperSize = Field(default=PaperSize.A4)
    transparent_background: bool = Field(
        default=False,
        description="If true, export with transparent background (PNG only).",
    )


# ============================================================
# Response Models
# ============================================================

class GenerateResponse(BaseModel):
    """Response body for POST /generate."""
    job_id: str
    ws_url: str = Field(description="WebSocket URL for real-time stroke streaming.")
    status: str = Field(default="queued")


class ExportResponse(BaseModel):
    """Response body for POST /export."""
    download_url: str
    format: ExportFormat
    file_size_bytes: int | None = None


class JobStatus(BaseModel):
    """Response body for GET /job/{job_id}."""
    job_id: str
    status: str = Field(description="queued | processing | complete | failed")
    progress: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Generation progress (0.0 to 1.0).",
    )
    error: str | None = None
