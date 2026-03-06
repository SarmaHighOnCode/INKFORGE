"""
INKFORGE — GET /styles

Returns available handwriting style presets.
"""

from fastapi import APIRouter

from app.models.schemas import StylePreset

router = APIRouter()


@router.get("/styles", response_model=list[StylePreset])
async def list_styles() -> list[StylePreset]:
    """Returns all available style presets."""
    return [
        StylePreset(
            id="neat_cursive", name="Neat Cursive", description="Elegant, connected cursive."
        ),
        StylePreset(id="casual_print", name="Casual Print", description="Clean block print."),
        StylePreset(id="rushed_notes", name="Rushed Notes", description="Messy, fast handwriting."),
        StylePreset(
            id="doctors_scrawl", name="Doctor's Scrawl", description="Barely legible scribbles."
        ),
        StylePreset(
            id="elegant_formal", name="Elegant Formal", description="Calligraphy-style writing."
        ),
    ]
