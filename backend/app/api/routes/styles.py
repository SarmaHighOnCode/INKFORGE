"""
INKFORGE — GET /styles

List all available preloaded handwriting style presets.
"""

from fastapi import APIRouter

from app.models.schemas import StylePreset

router = APIRouter()

# Preloaded style presets derived from clustered IAM training samples
STYLE_PRESETS: list[StylePreset] = [
    StylePreset(
        id="neat_cursive",
        name="Neat Cursive",
        description="Clean, flowing cursive handwriting with consistent letter connections.",
    ),
    StylePreset(
        id="casual_print",
        name="Casual Print",
        description="Relaxed print handwriting — clear, slightly irregular spacing.",
    ),
    StylePreset(
        id="rushed_notes",
        name="Rushed Notes",
        description="Quick, compressed handwriting with visible speed artifacts.",
    ),
    StylePreset(
        id="doctors_scrawl",
        name="Doctor's Scrawl",
        description="Highly compressed, barely legible — maximum inconsistency.",
    ),
    StylePreset(
        id="elegant_formal",
        name="Elegant Formal",
        description="Deliberate, well-spaced handwriting with slight calligraphic flair.",
    ),
]


@router.get("/styles", response_model=list[StylePreset])
async def list_styles() -> list[StylePreset]:
    """
    List all available handwriting style presets.

    Returns:
        List of StylePreset objects with id, name, and description.
    """
    return STYLE_PRESETS
