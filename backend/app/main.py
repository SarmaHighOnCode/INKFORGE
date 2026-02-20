"""
INKFORGE — FastAPI Application Entrypoint

This module initializes the FastAPI application, registers routes,
configures CORS, and sets up WebSocket support for real-time stroke streaming.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title="Inkforge API",
    description="Human-Like Handwriting Synthesis Engine — LSTM+MDN stroke-level generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
# TODO: Register route modules
# from app.api.routes import generate, export, styles, health
# app.include_router(generate.router, prefix="/api", tags=["generation"])
# app.include_router(export.router, prefix="/api", tags=["export"])
# app.include_router(styles.router, prefix="/api", tags=["styles"])
# app.include_router(health.router, tags=["health"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint — API info."""
    return {
        "name": "Inkforge API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }
