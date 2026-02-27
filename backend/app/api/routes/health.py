"""
INKFORGE — GET /health

LLM-aware service health check. Reports engine status, VRAM usage,
quantization config, and inference availability.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger("inkforge.routes.health")

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict:
    """
    Comprehensive health check endpoint.

    Returns:
        Service status, engine state, VRAM usage, quantization info,
        and GPU availability. Used by load balancers, monitoring,
        and the frontend to check if the backend is ready.
    """
    # Get engine from app state (set during lifespan)
    engine = getattr(request.app.state, "engine", None)

    if engine is None:
        return {
            "status": "starting",
            "model_loaded": False,
            "detail": "Engine not yet initialized",
        }

    try:
        status = engine.get_status()
    except Exception as e:
        logger.error(f"Engine status check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e),
            "engine": None,
            "gpu": None,
            "inference": None,
        }

    return {
        "status": "healthy" if status.model_loaded else "degraded",
        "model_loaded": status.model_loaded,
        "engine": {
            "backend": status.engine_backend,
            "model_name": status.model_name,
            "device": status.device,
            "quantization": status.quantization,
            "max_seq_len": status.max_seq_len,
        },
        "gpu": {
            "available": status.gpu_available,
            "name": status.gpu_name,
            "vram_total_gb": status.vram_total_gb,
            "vram_allocated_gb": status.vram_allocated_gb,
            "vram_kv_cache_gb": status.kv_cache_gb,
        },
        "inference": {
            "active_requests": status.active_requests,
            "total_requests_served": status.total_requests_served,
            "uptime_seconds": status.uptime_seconds,
        },
    }
