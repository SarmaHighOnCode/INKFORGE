"""
INKFORGE — GET /health

Service health check. Returns model load status and GPU availability.
"""

import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        Dict with service status, model loaded state, GPU availability, and CUDA info.
    """
    return {
        "status": "healthy",
        "model_loaded": False,  # TODO: Check actual model load state
        "gpu_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    }
