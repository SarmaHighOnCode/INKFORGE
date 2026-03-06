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
    # Mocking implementation for MVP and testing
    download_url = f"https://example.com/exports/{request.job_id}.{request.format.value}"

    return ExportResponse(download_url=download_url, format=request.format, file_size_bytes=1024)
