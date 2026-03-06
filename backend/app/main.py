"""
INKFORGE — FastAPI Application Entrypoint

Production-grade FastAPI application with:
    - Async lifespan management (model loaded once on startup)
    - Server-Sent Events (SSE) for real-time stroke streaming
    - CORS, route registration, health introspection
    - Ready for LLM / heavy Transformer model serving
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ml.llm_engine import EngineConfig, LLMEngine

# --- Logging Setup ---
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inkforge.app")


# ============================================================
# Lifespan — Load model ONCE on startup, release on shutdown
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    On startup:
        - Initialize the LLMEngine singleton
        - Load model weights into VRAM (or mock in dev mode)
        - Store engine reference on app.state for route access

    On shutdown:
        - Release VRAM
        - Clean up model resources

    This ensures the 15+ GB model is loaded ONCE, not per-request.
    """
    engine = LLMEngine.get_instance()

    config = EngineConfig(
        model_name=settings.model_name,
        checkpoint_path=settings.model_checkpoint_path,
        device=settings.device,
        engine_backend=settings.engine_backend,
        gpu_memory_fraction=settings.gpu_memory_fraction,
        quantization_bits=settings.quantization_bits,
        kv_cache_size_gb=settings.kv_cache_size_gb,
        max_seq_len=settings.max_seq_len,
        max_concurrent_requests=settings.max_concurrent_requests,
        stream_chunk_delay_ms=settings.stream_chunk_delay_ms,
    )

    try:
        await engine.initialize_model(config)
        app.state.engine = engine

        logger.info("FastAPI application ready — accepting requests")

        yield  # App runs here
    finally:
        # Shutdown — always runs even if initialize_model raised
        logger.info("Shutting down FastAPI application...")
        try:
            await engine.shutdown()
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")
        logger.info("Shutdown complete")


# ============================================================
# App Factory
# ============================================================

app = FastAPI(
    title="Inkforge API",
    description=(
        "Long-Form Human Handwriting Synthesis Engine — "
        "LLM-ready backend with streaming inference, VRAM management, "
        "and document-level stroke generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
from app.api.routes import export, generate, health  # noqa: E402

app.include_router(generate.router, prefix="/api", tags=["generation"])
app.include_router(health.router, tags=["health"])

# Optional: register export and styles routes when ready
app.include_router(export.router, prefix="/api", tags=["export"])
# app.include_router(styles.router, prefix="/api", tags=["styles"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint — API info."""
    return {
        "name": "Inkforge API",
        "version": "1.0.0",
        "status": "operational",
        "engine": settings.engine_backend,
        "docs": "/docs",
    }
