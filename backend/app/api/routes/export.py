"""
INKFORGE — POST /export

Re-render a completed generation job at print resolution (300 DPI).
Returns a signed download URL for PNG, PDF, or SVG.
"""

from fastapi import APIRouter

from app.models.schemas import ExportRequest, ExportResponse

router = APIRouter()


@router.post("/export", response_model=ExportResponse)
async def export_handwriting(request: ExportRequest) -> ExportResponse:
    """
    Export a completed generation job at print resolution.

    Args:
        request: ExportRequest with job_id, format, and paper settings.

    Returns:
        ExportResponse with download_url.
    """
    # TODO: Implement
    # 1. Retrieve completed stroke sequence from job_id
    # 2. Apply paper texture and ink color
    # 3. Render at 300 DPI via CairoSVG + Pillow
    # 4. Save to export storage
    # 5. Return signed download URL
    raise NotImplementedError("Export endpoint not yet implemented")
